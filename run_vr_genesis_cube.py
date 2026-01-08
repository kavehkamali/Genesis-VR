# run_vr_genesis_debug_colors_fixed.py
#
# Fixes GenesisException('RigidEntity is not built yet.') by ensuring we do NOT call
# set_pos() on entities until AFTER scene.build().
#
# Controls:
#   - Hold Grip (side/squeeze) -> move ACTIVE cube with right-hand pose
#   - Button A -> RED
#   - Button B -> BLUE
#   - Trigger > 0.2 -> GREEN
#   - Thumbstick moved beyond deadzone -> YELLOW
#   - Else -> GRAY
#
# Debug: prints changes for grip/trigger/A/B/thumbstick and pose valid.

import ctypes
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import xr  # pyopenxr

# SDL2 / OpenGL
if platform.system() == "Windows":
    SDL2_DIR = r"C:\\SDL2"
    if os.path.isdir(SDL2_DIR):
        os.add_dll_directory(SDL2_DIR)

import sdl2
from OpenGL import GL
from OpenGL import WGL


# ==========================================================
# USER TUNABLES (keep your working setup)
# ==========================================================
WORLD_T = (0.0, -1.2, 0.0)
MAPPING = "A"
FLIP_IMAGE_VERTICAL = True

RENDER_SCALE = 1.0
EYE_SEP_SCALE = 7.0
SWAP_EYES = False

STICK_DEADZONE = 0.30
DEBUG_INPUTS = True


# ==========================================================
# Shared memory: TRIPLE buffer per eye
# ==========================================================
@dataclass
class ShmStereoTB:
    L: List[str]
    R: List[str]
    w: int
    h: int
    channels: int = 3


def shm_create_triple(w: int, h: int):
    nbytes = w * h * 3
    shms = [SharedMemory(create=True, size=nbytes) for _ in range(6)]
    meta = ShmStereoTB(
        L=[shms[0].name, shms[1].name, shms[2].name],
        R=[shms[3].name, shms[4].name, shms[5].name],
        w=w,
        h=h,
    )
    return meta, shms


def shm_attach_triple(meta: ShmStereoTB):
    return [SharedMemory(name=n) for n in (meta.L + meta.R)]


def shm_view_rgb(shm: SharedMemory, w: int, h: int) -> np.ndarray:
    return np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)


# ==========================================================
# Math helpers
# ==========================================================
def quat_to_rotmat_xyzw(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def xr_basis_from_quat(q_xyzw):
    # OpenXR: right=+X, up=+Y, forward=-Z
    qx, qy, qz, qw = q_xyzw
    R = quat_to_rotmat_xyzw(qx, qy, qz, qw)
    right = R[:, 0]
    up = R[:, 1]
    forward = -R[:, 2]
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def normalize(v):
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def map_vec_xr_to_gen(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    if MAPPING == "A":
        return np.array([x, -z, y], dtype=np.float32)
    if MAPPING == "B":
        return np.array([x, z, y], dtype=np.float32)
    return np.array([x, y, -z], dtype=np.float32)


def map_pos_xr_to_gen(p_xyz, world_t=WORLD_T):
    p = map_vec_xr_to_gen(p_xyz)
    return (float(p[0] + world_t[0]), float(p[1] + world_t[1]), float(p[2] + world_t[2]))


# ==========================================================
# Genesis worker process
# ==========================================================
def _force_utf8_stdio():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:
        pass


def _to_uint8_rgb(img):
    if img is None:
        return None
    try:
        import torch
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)
    except Exception:
        arr = np.asarray(img)

    while arr.ndim > 3:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return arr


def _render_rgb(cam):
    out = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    if isinstance(out, dict):
        rgb = out.get("rgb", None)
    elif isinstance(out, (tuple, list)):
        rgb = out[0]
    else:
        rgb = out
    return _to_uint8_rgb(rgb)


def _cam_set_pose(cam, pos, lookat, up):
    try:
        cam.set_pose(pos=pos, lookat=lookat, up=up)
        return
    except TypeError:
        pass
    cam.set_pose(pos=pos, lookat=lookat)


def _make_surface(gs, rgb):
    try:
        s = gs.surfaces.Default()
        s.color = tuple(float(x) for x in rgb)
        return s
    except Exception:
        return None


def genesis_worker(meta: ShmStereoTB,
                   ctrl_queue: mp.Queue,
                   status_queue: mp.Queue,
                   published_idx: mp.Value,
                   published_seq: mp.Value,
                   stop_event: mp.Event,
                   vfov_deg: float):
    try:
        _force_utf8_stdio()

        os.environ.setdefault("GENESIS_LOG_LEVEL", "WARNING")

        import genesis as gs
        try:
            gs.init(backend=gs.cpu, logging_level="warning", theme="dumb")
        except TypeError:
            gs.init(backend=gs.cpu)

        shms = shm_attach_triple(meta)
        L = [shm_view_rgb(shms[i], meta.w, meta.h) for i in range(3)]
        R = [shm_view_rgb(shms[i + 3], meta.w, meta.h) for i in range(3)]

        scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer())
        scene.add_entity(gs.morphs.Plane())

        ring_z = 0.10
        ring_outer = 1.0
        hole = 0.35
        plate_t = 0.05
        rail_w = (ring_outer - hole) / 2.0

        scene.add_entity(gs.morphs.Box(size=(ring_outer, rail_w, plate_t),
                                       pos=(0.0, -(hole / 2.0 + rail_w / 2.0), ring_z),
                                       fixed=True))
        scene.add_entity(gs.morphs.Box(size=(ring_outer, rail_w, plate_t),
                                       pos=(0.0, +(hole / 2.0 + rail_w / 2.0), ring_z),
                                       fixed=True))
        scene.add_entity(gs.morphs.Box(size=(rail_w, hole, plate_t),
                                       pos=(-(hole / 2.0 + rail_w / 2.0), 0.0, ring_z),
                                       fixed=True))
        scene.add_entity(gs.morphs.Box(size=(rail_w, hole, plate_t),
                                       pos=(+(hole / 2.0 + rail_w / 2.0), 0.0, ring_z),
                                       fixed=True))

        # Create all cubes at START_POS (no set_pos yet!)
        cube_size = (0.08, 0.08, 0.08)
        start_pos = (0.30, 0.0, ring_z + 0.25)
        hide_pos = (0.0, 0.0, -10.0)

        COLORS = {
            "gray": (0.6, 0.6, 0.6),
            "red": (1.0, 0.1, 0.1),
            "blue": (0.1, 0.3, 1.0),
            "green": (0.2, 1.0, 0.2),
            "yellow": (1.0, 1.0, 0.2),
        }
        color_names = ["gray", "red", "blue", "green", "yellow"]

        cubes = []
        for name in color_names:
            surf = _make_surface(gs, COLORS[name])
            if surf is not None:
                ent = scene.add_entity(gs.morphs.Box(size=cube_size, pos=start_pos, fixed=False), surface=surf)
            else:
                ent = scene.add_entity(gs.morphs.Box(size=cube_size, pos=start_pos, fixed=False))
            cubes.append(ent)

        cam = [
            scene.add_camera(res=(meta.w, meta.h), pos=(1.2, -0.03, 1.0), lookat=(0, 0, ring_z), fov=vfov_deg, GUI=False),
            scene.add_camera(res=(meta.w, meta.h), pos=(1.2, +0.03, 1.0), lookat=(0, 0, ring_z), fov=vfov_deg, GUI=False),
        ]

        # BUILD FIRST
        scene.build()

        # Now it is safe to move/hide/switch
        active_idx = 0  # gray
        for i in range(len(cubes)):
            if i == active_idx:
                cubes[i].set_pos(start_pos, zero_velocity=True)
            else:
                cubes[i].set_pos(hide_pos, zero_velocity=True)

        status_queue.put({"ok": True, "msg": "Genesis worker ready"})

        want_seq = -1
        next_write = 0
        last_cube_pos = start_pos

        while not stop_event.is_set():
            try:
                msg = ctrl_queue.get(timeout=0.05)
            except Exception:
                continue
            if msg is None:
                continue

            seq = int(msg.get("seq", -1))
            if seq == want_seq:
                continue
            want_seq = seq

            grip = bool(msg.get("grip", False))
            trigger = float(msg.get("trigger", 0.0))
            button_a = bool(msg.get("button_a", False))
            button_b = bool(msg.get("button_b", False))
            stick_list = msg.get("stick", [0.0, 0.0])
            sx, sy = float(stick_list[0]), float(stick_list[1])

            hp = msg.get("hand_pos", None)
            hand_pos = tuple(hp) if hp is not None else None
            eye_pose = msg.get("eye_pose", [None, None])

            stick_active = (abs(sx) > STICK_DEADZONE) or (abs(sy) > STICK_DEADZONE)

            desired_name = "gray"
            if button_a:
                desired_name = "red"
            elif button_b:
                desired_name = "blue"
            elif stick_active:
                desired_name = "yellow"
            elif trigger > 0.2:
                desired_name = "green"
            desired_idx = color_names.index(desired_name)

            if desired_idx != active_idx:
                cubes[active_idx].set_pos(hide_pos, zero_velocity=True)
                active_idx = desired_idx
                cubes[active_idx].set_pos(last_cube_pos, zero_velocity=True)

            if grip and hand_pos is not None:
                hx, hy, hz = hand_pos
                gx, gy, gz = map_pos_xr_to_gen((hx, hy, hz), world_t=WORLD_T)
                x = float(np.clip(gx, -0.45, 0.45))
                y = float(np.clip(gy, -0.45, 0.45))
                z = float(ring_z + 0.12 + 0.25 * trigger)
                last_cube_pos = (x, y, z)
                cubes[active_idx].set_pos(last_cube_pos, zero_velocity=True)

            scene.step()

            # Eye separation scaling
            head = None
            if eye_pose[0] is not None and eye_pose[1] is not None:
                p0 = np.array(eye_pose[0]["p"], dtype=np.float32)
                p1 = np.array(eye_pose[1]["p"], dtype=np.float32)
                head = 0.5 * (p0 + p1)

            for i in (0, 1):
                if eye_pose[i] is None:
                    continue
                px, py, pz = eye_pose[i]["p"]
                qx, qy, qz, qw = eye_pose[i]["q"]

                if head is not None:
                    pi = np.array([px, py, pz], dtype=np.float32)
                    pi = head + EYE_SEP_SCALE * (pi - head)
                    px, py, pz = float(pi[0]), float(pi[1]), float(pi[2])

                _, up, forward = xr_basis_from_quat((qx, qy, qz, qw))
                g_forward = normalize(map_vec_xr_to_gen(forward))
                g_up = normalize(map_vec_xr_to_gen(up))

                gpos = map_pos_xr_to_gen((px, py, pz), world_t=WORLD_T)
                lookat = (gpos[0] + float(g_forward[0]),
                          gpos[1] + float(g_forward[1]),
                          gpos[2] + float(g_forward[2]))
                gup = (float(g_up[0]), float(g_up[1]), float(g_up[2]))

                _cam_set_pose(cam[i], pos=gpos, lookat=lookat, up=gup)

            write_idx = next_write % 3
            next_write += 1

            rgb0 = _render_rgb(cam[0])
            rgb1 = _render_rgb(cam[1])

            if FLIP_IMAGE_VERTICAL:
                if rgb0 is not None:
                    rgb0 = rgb0[::-1].copy()
                if rgb1 is not None:
                    rgb1 = rgb1[::-1].copy()

            if rgb0 is not None:
                L[write_idx][:] = rgb0[: meta.h, : meta.w, :3]
            if rgb1 is not None:
                R[write_idx][:] = rgb1[: meta.h, : meta.w, :3]

            with published_idx.get_lock():
                published_idx.value = write_idx
            with published_seq.get_lock():
                published_seq.value = want_seq

        try:
            scene.destroy()
        except Exception:
            pass
        for s in shms:
            s.close()

    except Exception as e:
        try:
            status_queue.put({"ok": False, "msg": repr(e)})
        except Exception:
            pass


# ==========================================================
# OpenGL blit
# ==========================================================
def compile_shader(src: str, shader_type):
    sid = GL.glCreateShader(shader_type)
    GL.glShaderSource(sid, src)
    GL.glCompileShader(sid)
    ok = GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS)
    if not ok:
        raise RuntimeError(GL.glGetShaderInfoLog(sid).decode("utf-8", "ignore"))
    return sid


def link_program(vs, fs):
    pid = GL.glCreateProgram()
    GL.glAttachShader(pid, vs)
    GL.glAttachShader(pid, fs)
    GL.glLinkProgram(pid)
    ok = GL.glGetProgramiv(pid, GL.GL_LINK_STATUS)
    if not ok:
        raise RuntimeError(GL.glGetProgramInfoLog(pid).decode("utf-8", "ignore"))
    return pid


class FullscreenBlitter:
    def __init__(self, tex_w: int, tex_h: int):
        self.tw, self.th = tex_w, tex_h
        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, tex_w, tex_h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        vs_src = r"""
        #version 330 core
        out vec2 vUV;
        void main() {
            vec2 pos;
            if (gl_VertexID == 0) { pos = vec2(-1.0, -1.0); vUV = vec2(0.0, 0.0); }
            if (gl_VertexID == 1) { pos = vec2( 3.0, -1.0); vUV = vec2(2.0, 0.0); }
            if (gl_VertexID == 2) { pos = vec2(-1.0,  3.0); vUV = vec2(0.0, 2.0); }
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """
        fs_src = r"""
        #version 330 core
        in vec2 vUV;
        uniform sampler2D uTex;
        out vec4 oColor;
        void main() {
            vec3 c = texture(uTex, vUV).rgb;
            oColor = vec4(c, 1.0);
        }
        """
        vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
        fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
        self.prog = link_program(vs, fs)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        self.vao = GL.glGenVertexArrays(1)
        self.uTex = GL.glGetUniformLocation(self.prog, "uTex")

    def upload(self, rgb: np.ndarray):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.tw, self.th, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def draw(self):
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.prog)
        GL.glBindVertexArray(self.vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glUniform1i(self.uTex, 0)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


# ==========================================================
# OpenXR runtime (main process)
# ==========================================================
class XrRuntime:
    def __init__(self, title="VR Genesis Debug Colors FIXED"):
        if platform.system() != "Windows":
            raise RuntimeError("Windows-only (WGL + SDL).")

        exts = xr.enumerate_instance_extension_properties()
        avail = set(
            (e.extension_name.decode("utf-8", "ignore") if isinstance(e.extension_name, bytes) else e.extension_name)
            for e in exts
        )
        req = xr.KHR_OPENGL_ENABLE_EXTENSION_NAME
        req = req.decode("utf-8", "ignore") if isinstance(req, bytes) else req
        if req not in avail:
            raise RuntimeError("XR_KHR_opengl_enable not available")

        app_info = xr.ApplicationInfo("vr_genesis_debug_colors_fixed", 1, "pyopenxr", 0, xr.XR_CURRENT_API_VERSION)
        self.instance = xr.create_instance(xr.InstanceCreateInfo(application_info=app_info, enabled_extension_names=[req]))

        self.system_id = xr.get_system(self.instance, xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self.view_config_views = xr.enumerate_view_configuration_views(self.instance, self.system_id, self.view_config_type)

        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError("SDL_Init failed")

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 0)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.SDL_GL_CONTEXT_PROFILE_CORE)

        ww = self.view_config_views[0].recommended_image_rect_width // 2
        wh = self.view_config_views[0].recommended_image_rect_height // 2

        self.window = sdl2.SDL_CreateWindow(
            title.encode("utf-8"),
            sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED,
            ww, wh,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_SHOWN,
        )
        if not self.window:
            raise RuntimeError("Failed to create SDL window")

        self.gl_context = sdl2.SDL_GL_CreateContext(self.window)
        if not self.gl_context:
            raise RuntimeError("Failed to create OpenGL context")
        sdl2.SDL_GL_MakeCurrent(self.window, self.gl_context)
        sdl2.SDL_GL_SetSwapInterval(0)

        # REQUIRED: xrGetOpenGLGraphicsRequirementsKHR BEFORE xrCreateSession
        pfn_get_gl_reqs = ctypes.cast(
            xr.get_instance_proc_addr(self.instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )
        gl_reqs = xr.GraphicsRequirementsOpenGLKHR()
        res = pfn_get_gl_reqs(self.instance, self.system_id, ctypes.byref(gl_reqs))
        res = xr.exception.check_result(xr.Result(res))
        if res.is_exception():
            raise res

        gb = xr.GraphicsBindingOpenGLWin32KHR()
        gb.h_dc = WGL.wglGetCurrentDC()
        gb.h_glrc = WGL.wglGetCurrentContext()
        if not gb.h_dc or not gb.h_glrc:
            raise RuntimeError("Could not get current HDC/HGLRC from WGL.")
        gb_ptr = ctypes.cast(ctypes.pointer(gb), ctypes.c_void_p)

        self.session = xr.create_session(self.instance, xr.SessionCreateInfo(0, self.system_id, next=gb_ptr))

        identity = xr.Posef(orientation=xr.Quaternionf(0, 0, 0, 1), position=xr.Vector3f(0, 0, 0))

        # Prefer STAGE if available
        try:
            self.reference_space = xr.create_reference_space(
                self.session,
                xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.STAGE,
                                            pose_in_reference_space=identity),
            )
        except Exception:
            self.reference_space = xr.create_reference_space(
                self.session,
                xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.LOCAL,
                                            pose_in_reference_space=identity),
            )

        # Swapchains
        self.swapchains = []
        self.swapchain_images = []
        self.image_sizes = []

        formats = xr.enumerate_swapchain_formats(self.session)
        color_format = formats[0]
        for cand in (GL.GL_SRGB8_ALPHA8, GL.GL_RGBA8):
            if cand in formats:
                color_format = cand
                break

        for v in self.view_config_views:
            w, h = v.recommended_image_rect_width, v.recommended_image_rect_height
            sc = xr.create_swapchain(
                self.session,
                xr.SwapchainCreateInfo(
                    create_flags=0,
                    usage_flags=(xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT),
                    format=color_format,
                    sample_count=(v.recommended_swapchain_sample_count or 1),
                    width=w, height=h,
                    face_count=1, array_size=1, mip_count=1,
                ),
            )
            imgs = xr.enumerate_swapchain_images(sc, xr.SwapchainImageOpenGLKHR)
            self.swapchains.append(sc)
            self.swapchain_images.append(imgs)
            self.image_sizes.append((w, h))

        self.fbo = GL.glGenFramebuffers(1)

        # Actions
        self.action_set = xr.create_action_set(
            self.instance,
            xr.ActionSetCreateInfo(action_set_name="teleop", localized_action_set_name="Teleop", priority=0),
        )
        self.right_hand_path = xr.string_to_path(self.instance, "/user/hand/right")

        self.grip_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(action_name="right_grip", action_type=xr.ActionType.BOOLEAN_INPUT,
                                localized_action_name="Right Grip", subaction_paths=[self.right_hand_path]),
        )
        self.trigger_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(action_name="right_trigger", action_type=xr.ActionType.FLOAT_INPUT,
                                localized_action_name="Right Trigger", subaction_paths=[self.right_hand_path]),
        )
        self.hand_pose_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(action_name="right_hand_pose", action_type=xr.ActionType.POSE_INPUT,
                                localized_action_name="Right Hand Pose", subaction_paths=[self.right_hand_path]),
        )
        self.button_a_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(action_name="button_a", action_type=xr.ActionType.BOOLEAN_INPUT,
                                localized_action_name="Button A", subaction_paths=[self.right_hand_path]),
        )
        self.button_b_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(action_name="button_b", action_type=xr.ActionType.BOOLEAN_INPUT,
                                localized_action_name="Button B", subaction_paths=[self.right_hand_path]),
        )
        self.thumbstick_x_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(action_name="thumbstick_x", action_type=xr.ActionType.FLOAT_INPUT,
                                localized_action_name="Thumbstick X", subaction_paths=[self.right_hand_path]),
        )
        self.thumbstick_y_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(action_name="thumbstick_y", action_type=xr.ActionType.FLOAT_INPUT,
                                localized_action_name="Thumbstick Y", subaction_paths=[self.right_hand_path]),
        )

        profile = xr.string_to_path(self.instance, "/interaction_profiles/oculus/touch_controller")
        xr.suggest_interaction_profile_bindings(
            self.instance,
            xr.InteractionProfileSuggestedBinding(
                interaction_profile=profile,
                suggested_bindings=[
                    xr.ActionSuggestedBinding(self.grip_action, xr.string_to_path(self.instance, "/user/hand/right/input/squeeze/value")),
                    xr.ActionSuggestedBinding(self.trigger_action, xr.string_to_path(self.instance, "/user/hand/right/input/trigger/value")),
                    xr.ActionSuggestedBinding(self.hand_pose_action, xr.string_to_path(self.instance, "/user/hand/right/input/grip/pose")),
                    xr.ActionSuggestedBinding(self.button_a_action, xr.string_to_path(self.instance, "/user/hand/right/input/a/click")),
                    xr.ActionSuggestedBinding(self.button_b_action, xr.string_to_path(self.instance, "/user/hand/right/input/b/click")),
                    xr.ActionSuggestedBinding(self.thumbstick_x_action, xr.string_to_path(self.instance, "/user/hand/right/input/thumbstick/x")),
                    xr.ActionSuggestedBinding(self.thumbstick_y_action, xr.string_to_path(self.instance, "/user/hand/right/input/thumbstick/y")),
                ],
            ),
        )
        xr.attach_session_action_sets(self.session, xr.SessionActionSetsAttachInfo(action_sets=[self.action_set]))

        self.hand_space = xr.create_action_space(
            self.session,
            xr.ActionSpaceCreateInfo(action=self.hand_pose_action, subaction_path=self.right_hand_path,
                                     pose_in_action_space=identity),
        )

        xr.begin_session(self.session, xr.SessionBeginInfo(primary_view_configuration_type=self.view_config_type))
        self.active_action_set = xr.ActiveActionSet(action_set=self.action_set, subaction_path=xr.NULL_PATH)
        self.environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE

        self._dbg_last = {"grip": None, "a": None, "b": None, "trig": None, "sx": None, "sy": None, "pose_valid": None}

    def shutdown(self):
        try: xr.end_session(self.session)
        except Exception: pass
        try: xr.destroy_session(self.session)
        except Exception: pass
        try: xr.destroy_instance(self.instance)
        except Exception: pass
        try: GL.glDeleteFramebuffers(1, [self.fbo])
        except Exception: pass
        try: sdl2.SDL_GL_DeleteContext(self.gl_context)
        except Exception: pass
        try: sdl2.SDL_DestroyWindow(self.window)
        except Exception: pass
        try: sdl2.SDL_Quit()
        except Exception: pass

    def _dbg_print_changes(self, grip, trigger, a, b, sx, sy, pose_valid):
        if not DEBUG_INPUTS:
            return
        last = self._dbg_last

        def pr(name, val):
            print(f"[DBG] {name}: {val}")

        if last["grip"] is None or grip != last["grip"]:
            pr("Grip", int(grip))
        if last["a"] is None or a != last["a"]:
            pr("Button A", int(a))
        if last["b"] is None or b != last["b"]:
            pr("Button B", int(b))

        if last["trig"] is None or abs(trigger - last["trig"]) > 0.05:
            pr("Trigger", f"{trigger:.2f}")

        if last["sx"] is None or abs(sx - last["sx"]) > 0.20 or (abs(sx) < 0.05 and abs(last["sx"]) >= 0.05):
            pr("Thumbstick X", f"{sx:.2f}")
        if last["sy"] is None or abs(sy - last["sy"]) > 0.20 or (abs(sy) < 0.05 and abs(last["sy"]) >= 0.05):
            pr("Thumbstick Y", f"{sy:.2f}")

        if last["pose_valid"] is None or pose_valid != last["pose_valid"]:
            pr("Hand pose valid", int(pose_valid))

        last["grip"] = grip
        last["a"] = a
        last["b"] = b
        last["trig"] = trigger
        last["sx"] = sx
        last["sy"] = sy
        last["pose_valid"] = pose_valid

    def run(self, get_tb_frames, send_to_worker, blit_left: FullscreenBlitter, blit_right: FullscreenBlitter):
        running = True
        sdl_event = sdl2.SDL_Event()
        seq = 0

        while running:
            while sdl2.SDL_PollEvent(ctypes.byref(sdl_event)) != 0:
                if sdl_event.type == sdl2.SDL_QUIT:
                    running = False
                    break
                if sdl_event.type == sdl2.SDL_KEYDOWN and sdl_event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    running = False
                    break
            if not running:
                break

            while True:
                try:
                    xr.poll_event(self.instance)
                except xr.exception.EventUnavailable:
                    break

            frame_state = xr.wait_frame(self.session)
            xr.begin_frame(self.session)

            try:
                xr.sync_actions(self.session, xr.ActionsSyncInfo(active_action_sets=[self.active_action_set]))
            except Exception:
                pass

            grip = False
            trigger = 0.0
            button_a = False
            button_b = False
            sx, sy = 0.0, 0.0
            hand_pos = None
            pose_valid = False

            try:
                g = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(action=self.grip_action, subaction_path=xr.NULL_PATH))
                t = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(action=self.trigger_action, subaction_path=xr.NULL_PATH))
                a = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(action=self.button_a_action, subaction_path=xr.NULL_PATH))
                b = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(action=self.button_b_action, subaction_path=xr.NULL_PATH))
                ax = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(action=self.thumbstick_x_action, subaction_path=xr.NULL_PATH))
                ay = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(action=self.thumbstick_y_action, subaction_path=xr.NULL_PATH))

                grip = bool(g.current_state)
                trigger = float(t.current_state)
                button_a = bool(a.current_state)
                button_b = bool(b.current_state)
                sx, sy = float(ax.current_state), float(ay.current_state)

                loc = xr.locate_space(self.hand_space, self.reference_space, frame_state.predicted_display_time)
                flags = loc.location_flags
                pose_valid = bool((flags & xr.SpaceLocationFlags.POSITION_VALID_BIT) and (flags & xr.SpaceLocationFlags.ORIENTATION_VALID_BIT))
                if pose_valid:
                    p = loc.pose.position
                    hand_pos = (float(p.x), float(p.y), float(p.z))
            except Exception:
                pass

            self._dbg_print_changes(grip, trigger, button_a, button_b, sx, sy, pose_valid)

            try:
                _, located_views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(
                        view_configuration_type=self.view_config_type,
                        display_time=frame_state.predicted_display_time,
                        space=self.reference_space
                    ),
                )
            except Exception:
                xr.end_frame(self.session, xr.FrameEndInfo(display_time=frame_state.predicted_display_time,
                                                           environment_blend_mode=self.environment_blend_mode,
                                                           layers=[]))
                continue

            eye_pose = []
            for i in (0, 1):
                p = located_views[i].pose.position
                q = located_views[i].pose.orientation
                eye_pose.append({"p": (float(p.x), float(p.y), float(p.z)),
                                 "q": (float(q.x), float(q.y), float(q.z), float(q.w))})

            seq += 1
            send_to_worker(
                seq=seq,
                grip=grip,
                trigger=trigger,
                button_a=button_a,
                button_b=button_b,
                stick=[sx, sy],
                hand_pos=list(hand_pos) if hand_pos is not None else None,
                eye_pose=eye_pose,
            )

            rgb_left, rgb_right = get_tb_frames(seq)
            if SWAP_EYES:
                rgb_left, rgb_right = rgb_right, rgb_left

            proj_views = []
            for eye_index, sc in enumerate(self.swapchains):
                w, h = self.image_sizes[eye_index]

                img_idx = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
                xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
                gl_image = self.swapchain_images[eye_index][img_idx].image

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, gl_image, 0)
                GL.glViewport(0, 0, w, h)

                if eye_index == 0:
                    blit_left.upload(rgb_left)
                    blit_left.draw()
                else:
                    blit_right.upload(rgb_right)
                    blit_right.draw()

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
                xr.release_swapchain_image(sc, xr.SwapchainImageReleaseInfo())

                sub_image = xr.SwapchainSubImage(
                    swapchain=sc,
                    image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(w, h)),
                    image_array_index=0,
                )
                proj_views.append(
                    xr.CompositionLayerProjectionView(
                        pose=located_views[eye_index].pose,
                        fov=located_views[eye_index].fov,
                        sub_image=sub_image,
                    )
                )

            proj_layer = xr.CompositionLayerProjection(
                layer_flags=xr.CompositionLayerFlags.NONE,
                space=self.reference_space,
                views=proj_views,
            )

            xr.end_frame(
                self.session,
                xr.FrameEndInfo(display_time=frame_state.predicted_display_time,
                                environment_blend_mode=self.environment_blend_mode,
                                layers=[ctypes.pointer(proj_layer)]),
            )


# ==========================================================
# Main
# ==========================================================
def main():
    mp.set_start_method("spawn", force=True)

    rt = XrRuntime("VR Genesis Debug Colors FIXED")

    sw_w, sw_h = rt.image_sizes[0]
    render_w = max(64, int(sw_w * RENDER_SCALE))
    render_h = max(64, int(sw_h * RENDER_SCALE))

    print("[Config]")
    print("  WORLD_T =", WORLD_T, "MAPPING =", MAPPING, "FLIP_IMAGE_VERTICAL =", FLIP_IMAGE_VERTICAL)
    print("  RENDER_SCALE =", RENDER_SCALE, "render =", (render_w, render_h), "swapchain =", (sw_w, sw_h))
    print("  EYE_SEP_SCALE =", EYE_SEP_SCALE, "SWAP_EYES =", SWAP_EYES)
    print("  DEBUG_INPUTS =", DEBUG_INPUTS)

    meta, shms_main = shm_create_triple(render_w, render_h)
    L = [shm_view_rgb(shms_main[i], render_w, render_h) for i in range(3)]
    R = [shm_view_rgb(shms_main[i + 3], render_w, render_h) for i in range(3)]

    ctrl_queue = mp.Queue(maxsize=1)
    status_queue = mp.Queue(maxsize=10)
    stop_event = mp.Event()

    published_idx = mp.Value("i", 0)
    published_seq = mp.Value("i", 0)

    vfov_deg = 92.0

    p = mp.Process(
        target=genesis_worker,
        args=(meta, ctrl_queue, status_queue, published_idx, published_seq, stop_event, vfov_deg),
        daemon=True
    )
    p.start()

    ok = False
    t0 = time.time()
    while time.time() - t0 < 30:
        try:
            msg = status_queue.get_nowait()
            print("[Worker]", msg)
            ok = bool(msg.get("ok", False))
            break
        except Exception:
            time.sleep(0.05)
    if not ok:
        stop_event.set()
        p.join(timeout=2)
        raise RuntimeError("Worker failed to start")

    def send_to_worker(**payload):
        try:
            if ctrl_queue.full():
                _ = ctrl_queue.get_nowait()
            ctrl_queue.put_nowait(payload)
        except Exception:
            pass

    left_local = np.empty((render_h, render_w, 3), dtype=np.uint8)
    right_local = np.empty((render_h, render_w, 3), dtype=np.uint8)

    def get_tb_frames(expected_seq: int):
        for _ in range(5):
            with published_seq.get_lock():
                ps = int(published_seq.value)
            if ps >= expected_seq:
                break
            time.sleep(0.001)

        with published_idx.get_lock():
            idx = int(published_idx.value)

        np.copyto(left_local, L[idx])
        np.copyto(right_local, R[idx])
        return left_local, right_local

    blit_left = FullscreenBlitter(render_w, render_h)
    blit_right = FullscreenBlitter(render_w, render_h)

    try:
        rt.run(get_tb_frames=get_tb_frames, send_to_worker=send_to_worker,
               blit_left=blit_left, blit_right=blit_right)
    finally:
        stop_event.set()
        try:
            p.join(timeout=3)
        except Exception:
            pass

        for s in shms_main:
            try:
                s.close()
            except Exception:
                pass
        for name in meta.L + meta.R:
            try:
                SharedMemory(name=name).unlink()
            except Exception:
                pass

        rt.shutdown()


if __name__ == "__main__":
    main()
