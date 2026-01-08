import ctypes
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
# USER TUNABLES
# ==========================================================
WORLD_T = (0.0, -1.2, 0.0)          # your working placement
MAPPING = "A"                       # your working mapping
FLIP_IMAGE_VERTICAL = True

RENDER_SCALE = 1.0                  # 0.6–1.0 (lower = faster, less lag)
TARGET_RENDER_HZ = 72               # worker render pacing (72/80/90)

EYE_SEP_SCALE = 7.0                 # stereo comfort: try 0.9..1.1, NOT 7.0
SWAP_EYES = False

# smoothing (bigger = smoother but more lag)
POSE_SMOOTH_POS = 0.65              # 0..1  (0=no smoothing, 0.6..0.8 good)
POSE_SMOOTH_ROT = 0.65              # 0..1

# reduce Genesis logs
SILENCE_GENESIS_LOGS = True


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


def shm_create_triple(w: int, h: int) -> Tuple[ShmStereoTB, List[SharedMemory]]:
    nbytes = w * h * 3
    shms = [SharedMemory(create=True, size=nbytes) for _ in range(6)]
    meta = ShmStereoTB(
        L=[shms[0].name, shms[1].name, shms[2].name],
        R=[shms[3].name, shms[4].name, shms[5].name],
        w=w, h=h
    )
    return meta, shms


def shm_attach_triple(meta: ShmStereoTB) -> List[SharedMemory]:
    return [SharedMemory(name=n) for n in (meta.L + meta.R)]


def shm_view_rgb(shm: SharedMemory, w: int, h: int) -> np.ndarray:
    return np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)


# ==========================================================
# Math helpers
# ==========================================================

def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-9)


def quat_dot(a, b):
    return float(np.dot(a, b))


def quat_slerp(q0, q1, t: float):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    # shortest path
    if quat_dot(q0, q1) < 0.0:
        q1 = -q1

    dot = np.clip(quat_dot(q0, q1), -1.0, 1.0)

    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / math.sin(theta_0)
    s1 = math.sin(theta) / math.sin(theta_0)
    return quat_normalize((s0 * q0) + (s1 * q1))


def quat_to_rotmat_xyzw(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
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
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


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


def genesis_worker(meta: ShmStereoTB,
                   ctrl_queue: mp.Queue,
                   status_queue: mp.Queue,
                   published_idx: mp.Value,
                   published_seq: mp.Value,
                   stop_event: mp.Event,
                   vfov_deg: float):
    try:
        _force_utf8_stdio()

        if SILENCE_GENESIS_LOGS:
            import logging
            logging.getLogger().handlers.clear()
            logging.getLogger().setLevel(logging.ERROR)
            for name in ["genesis", "gstaichi", "taichi"]:
                logging.getLogger(name).setLevel(logging.ERROR)

        import genesis as gs

        # IMPORTANT: Genesis supports logging_level in gs.init() :contentReference[oaicite:2]{index=2}
        if SILENCE_GENESIS_LOGS:
            gs.init(backend=gs.cpu, logging_level="error")
        else:
            gs.init(backend=gs.cpu)

        shms = shm_attach_triple(meta)
        L = [shm_view_rgb(shms[i], meta.w, meta.h) for i in range(3)]
        R = [shm_view_rgb(shms[i+3], meta.w, meta.h) for i in range(3)]

        scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer())

        # Ring plate + cube
        scene.add_entity(gs.morphs.Plane())

        ring_z = 0.10
        ring_outer = 1.0
        hole = 0.35
        plate_t = 0.05
        rail_w = (ring_outer - hole) / 2.0

        scene.add_entity(gs.morphs.Box(size=(ring_outer, rail_w, plate_t),
                                       pos=(0.0, -(hole/2.0 + rail_w/2.0), ring_z),
                                       fixed=True))
        scene.add_entity(gs.morphs.Box(size=(ring_outer, rail_w, plate_t),
                                       pos=(0.0, +(hole/2.0 + rail_w/2.0), ring_z),
                                       fixed=True))
        scene.add_entity(gs.morphs.Box(size=(rail_w, hole, plate_t),
                                       pos=(-(hole/2.0 + rail_w/2.0), 0.0, ring_z),
                                       fixed=True))
        scene.add_entity(gs.morphs.Box(size=(rail_w, hole, plate_t),
                                       pos=(+(hole/2.0 + rail_w/2.0), 0.0, ring_z),
                                       fixed=True))

        cube = scene.add_entity(gs.morphs.Box(size=(0.08, 0.08, 0.08),
                                              pos=(0.30, 0.0, ring_z + 0.25),
                                              fixed=False))

        camL = scene.add_camera(res=(meta.w, meta.h), pos=(1.2, -0.03, 1.0),
                                lookat=(0, 0, ring_z), fov=vfov_deg, GUI=False)
        camR = scene.add_camera(res=(meta.w, meta.h), pos=(1.2, +0.03, 1.0),
                                lookat=(0, 0, ring_z), fov=vfov_deg, GUI=False)

        scene.build()
        status_queue.put({"ok": True, "msg": "Genesis worker ready"})

        # latest inputs
        grip = False
        trigger = 0.0
        hand_pos = None

        head_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        head_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        ipd = 0.064  # default fallback meters

        have_pose = False
        want_seq = -1

        # smoothed state
        head_pos_s = head_pos.copy()
        head_q_s = head_q.copy()

        next_write = 0
        dt_target = 1.0 / max(10.0, float(TARGET_RENDER_HZ))
        t_next = time.perf_counter()

        while not stop_event.is_set():
            # non-blocking drain: ALWAYS render at steady rate (prevents "jump every 0.05s")
            got_any = False
            try:
                while True:
                    msg = ctrl_queue.get_nowait()
                    if msg is None:
                        continue
                    got_any = True
                    seq = int(msg.get("seq", -1))
                    if seq == want_seq:
                        continue
                    want_seq = seq

                    grip = bool(msg.get("grip", grip))
                    trigger = float(msg.get("trigger", trigger))

                    hp = msg.get("hand_pos", None)
                    hand_pos = tuple(hp) if hp is not None else None

                    hp2 = msg.get("head_pos", None)
                    hq2 = msg.get("head_q", None)
                    ipd2 = msg.get("ipd", None)

                    if hp2 is not None and hq2 is not None:
                        head_pos = np.array(hp2, dtype=np.float32)
                        head_q = np.array(hq2, dtype=np.float32)
                        have_pose = True
                    if ipd2 is not None:
                        ipd = float(ipd2)
            except Exception:
                pass

            # cube control
            if grip and hand_pos is not None:
                hx, hy, hz = hand_pos
                gx, gy, gz = map_pos_xr_to_gen((hx, hy, hz), world_t=WORLD_T)
                x = float(np.clip(gx, -0.45, 0.45))
                y = float(np.clip(gy, -0.45, 0.45))
                z = float(ring_z + 0.12 + 0.25 * trigger)
                cube.set_pos((x, y, z), zero_velocity=True)

            scene.step()

            # head pose smoothing (reduces jitter at cost of a bit of latency)
            if have_pose:
                # position EMA
                head_pos_s = (POSE_SMOOTH_POS * head_pos_s) + ((1.0 - POSE_SMOOTH_POS) * head_pos)

                # rotation slerp smoothing toward newest
                head_q_s = quat_slerp(head_q_s, head_q, 1.0 - POSE_SMOOTH_ROT)

                # build Genesis L/R cameras from ONE shared orientation + IPD baseline
                right, up, forward = xr_basis_from_quat(head_q_s)
                g_forward = normalize(map_vec_xr_to_gen(forward))
                g_up = normalize(map_vec_xr_to_gen(up))
                g_right = normalize(map_vec_xr_to_gen(right))

                # map head position to Genesis space
                g_head = np.array(map_pos_xr_to_gen(head_pos_s, world_t=WORLD_T), dtype=np.float32)

                half_sep = 0.5 * ipd * float(EYE_SEP_SCALE)
                g_L = g_head - g_right * half_sep
                g_R = g_head + g_right * half_sep

                lookL = g_L + g_forward
                lookR = g_R + g_forward

                _cam_set_pose(camL,
                             pos=(float(g_L[0]), float(g_L[1]), float(g_L[2])),
                             lookat=(float(lookL[0]), float(lookL[1]), float(lookL[2])),
                             up=(float(g_up[0]), float(g_up[1]), float(g_up[2])))

                _cam_set_pose(camR,
                             pos=(float(g_R[0]), float(g_R[1]), float(g_R[2])),
                             lookat=(float(lookR[0]), float(lookR[1]), float(lookR[2])),
                             up=(float(g_up[0]), float(g_up[1]), float(g_up[2])))

            # render
            write_idx = next_write % 3
            next_write += 1

            rgb0 = _render_rgb(camL)
            rgb1 = _render_rgb(camR)

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

            # steady pacing
            t_now = time.perf_counter()
            t_next += dt_target
            sleep_s = t_next - t_now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # we're late; reset to avoid runaway drift
                t_next = t_now

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
    def __init__(self, title="VR Genesis Stable"):
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

        app_info = xr.ApplicationInfo("vr_genesis_stable", 1, "pyopenxr", 0, xr.XR_CURRENT_API_VERSION)
        self.instance = xr.create_instance(
            xr.InstanceCreateInfo(application_info=app_info, enabled_extension_names=[req])
        )

        self.system_id = xr.get_system(self.instance, xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self.view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self.view_config_type
        )

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

        # Prefer STAGE for stable "world locked" if available
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

        # actions
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

        profile = xr.string_to_path(self.instance, "/interaction_profiles/oculus/touch_controller")
        xr.suggest_interaction_profile_bindings(
            self.instance,
            xr.InteractionProfileSuggestedBinding(
                interaction_profile=profile,
                suggested_bindings=[
                    xr.ActionSuggestedBinding(self.grip_action, xr.string_to_path(self.instance, "/user/hand/right/input/squeeze/value")),
                    xr.ActionSuggestedBinding(self.trigger_action, xr.string_to_path(self.instance, "/user/hand/right/input/trigger/value")),
                    xr.ActionSuggestedBinding(self.hand_pose_action, xr.string_to_path(self.instance, "/user/hand/right/input/grip/pose")),
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

            # --- controller ---
            grip = False
            trigger = 0.0
            hand_pos = None

            try:
                g = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(action=self.grip_action, subaction_path=xr.NULL_PATH))
                t = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(action=self.trigger_action, subaction_path=xr.NULL_PATH))
                grip = bool(g.current_state)
                trigger = float(t.current_state)

                loc = xr.locate_space(self.hand_space, self.reference_space, frame_state.predicted_display_time)
                flags = loc.location_flags
                if (flags & xr.SpaceLocationFlags.POSITION_VALID_BIT) and (flags & xr.SpaceLocationFlags.ORIENTATION_VALID_BIT):
                    p = loc.pose.position
                    hand_pos = (float(p.x), float(p.y), float(p.z))
            except Exception:
                pass

            # --- views ---
            try:
                _, located_views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(view_configuration_type=self.view_config_type,
                                      display_time=frame_state.predicted_display_time,
                                      space=self.reference_space),
                )
            except Exception:
                xr.end_frame(self.session, xr.FrameEndInfo(display_time=frame_state.predicted_display_time,
                                                           environment_blend_mode=self.environment_blend_mode,
                                                           layers=[]))
                continue

            # compute head pose + ipd from eyes
            p0 = located_views[0].pose.position
            p1 = located_views[1].pose.position
            q0 = located_views[0].pose.orientation
            q1 = located_views[1].pose.orientation

            eye0 = np.array([float(p0.x), float(p0.y), float(p0.z)], dtype=np.float32)
            eye1 = np.array([float(p1.x), float(p1.y), float(p1.z)], dtype=np.float32)
            head_pos = 0.5 * (eye0 + eye1)
            ipd = float(np.linalg.norm(eye1 - eye0))

            qq0 = np.array([float(q0.x), float(q0.y), float(q0.z), float(q0.w)], dtype=np.float32)
            qq1 = np.array([float(q1.x), float(q1.y), float(q1.z), float(q1.w)], dtype=np.float32)
            # average orientation (slerp 50/50)
            head_q = quat_slerp(qq0, qq1, 0.5)

            seq += 1
            send_to_worker(
                seq=seq,
                grip=grip,
                trigger=trigger,
                hand_pos=list(hand_pos) if hand_pos is not None else None,
                head_pos=[float(head_pos[0]), float(head_pos[1]), float(head_pos[2])],
                head_q=[float(head_q[0]), float(head_q[1]), float(head_q[2]), float(head_q[3])],
                ipd=ipd,
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

    # (optional) reduce unicode issues in main too
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    rt = XrRuntime("VR Genesis Stable")

    sw_w, sw_h = rt.image_sizes[0]
    render_w = max(64, int(sw_w * RENDER_SCALE))
    render_h = max(64, int(sw_h * RENDER_SCALE))

    print("[Config]")
    print("  WORLD_T =", WORLD_T, "MAPPING =", MAPPING, "FLIP_IMAGE_VERTICAL =", FLIP_IMAGE_VERTICAL)
    print("  RENDER_SCALE =", RENDER_SCALE, "render =", (render_w, render_h), "swapchain =", (sw_w, sw_h))
    print("  TARGET_RENDER_HZ =", TARGET_RENDER_HZ)
    print("  EYE_SEP_SCALE =", EYE_SEP_SCALE, "SWAP_EYES =", SWAP_EYES)
    print("  POSE_SMOOTH_POS =", POSE_SMOOTH_POS, "POSE_SMOOTH_ROT =", POSE_SMOOTH_ROT)

    meta, shms_main = shm_create_triple(render_w, render_h)
    L = [shm_view_rgb(shms_main[i], render_w, render_h) for i in range(3)]
    R = [shm_view_rgb(shms_main[i+3], render_w, render_h) for i in range(3)]

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

    # local copy buffers to avoid tearing
    left_local = np.empty((render_h, render_w, 3), dtype=np.uint8)
    right_local = np.empty((render_h, render_w, 3), dtype=np.uint8)

    def get_tb_frames(expected_seq: int):
        # wait briefly for worker to catch up (tiny), but don't stall long
        for _ in range(2):
            with published_seq.get_lock():
                ps = int(published_seq.value)
            if ps >= expected_seq:
                break
            time.sleep(0.0005)

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
