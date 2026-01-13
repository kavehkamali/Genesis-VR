# run_vr_opengl_game_style.py
#
# Quest Link / OpenXR "game-style" pipeline:
# - Render directly into OpenXR swapchain GL textures (GPU->GPU)
# - Use OpenXR per-eye pose + FOV to build view/projection
# - No Genesis cam.render(), no CPU readback, no glTexSubImage2D, no shared memory
#
# Controls:
#   - Hold Grip -> move ACTIVE cube with right-hand pose
#   - Button A -> RED
#   - Button B -> BLUE
#   - Trigger > 0.2 -> GREEN
#   - Thumbstick beyond deadzone -> YELLOW
#   - Else -> GRAY
#
# This is what VR games do. It will feel dramatically more stable on head motion.

import ctypes
import os
import platform
import sys
import time
import math

import numpy as np
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
# USER TUNABLES (match your original)
# ==========================================================
WORLD_T = (0.0, -1.2, 0.0)
MAPPING = "A"

EYE_SEP_SCALE = 7.0
SWAP_EYES = False

STICK_DEADZONE = 0.30
DEBUG_INPUTS = True

# Scene layout (same as your Genesis scene)
RING_Z = 0.10
RING_OUTER = 1.0
HOLE = 0.35
PLATE_T = 0.05
RAIL_W = (RING_OUTER - HOLE) / 2.0

CUBE_SIZE = 0.08
START_POS = np.array([0.30, 0.0, RING_Z + 0.25], dtype=np.float32)

COLORS = {
    "gray":   (0.6, 0.6, 0.6, 1.0),
    "red":    (1.0, 0.1, 0.1, 1.0),
    "blue":   (0.1, 0.3, 1.0, 1.0),
    "green":  (0.2, 1.0, 0.2, 1.0),
    "yellow": (1.0, 1.0, 0.2, 1.0),
}
COLOR_NAMES = ["gray", "red", "blue", "green", "yellow"]


# ==========================================================
# Math helpers (same mapping idea as your code)
# ==========================================================
def normalize(v):
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n

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

def map_vec_xr_to_gen(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    if MAPPING == "A":
        return np.array([x, -z, y], dtype=np.float32)
    if MAPPING == "B":
        return np.array([x,  z, y], dtype=np.float32)
    return np.array([x, y, -z], dtype=np.float32)

def map_pos_xr_to_gen(p_xyz, world_t=WORLD_T):
    p = map_vec_xr_to_gen(p_xyz)
    return np.array([p[0] + world_t[0], p[1] + world_t[1], p[2] + world_t[2]], dtype=np.float32)


def mat4_identity():
    return np.eye(4, dtype=np.float32)

def mat4_translate(t):
    M = mat4_identity()
    M[0, 3] = t[0]
    M[1, 3] = t[1]
    M[2, 3] = t[2]
    return M

def mat4_scale(sx, sy, sz):
    M = mat4_identity()
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M

def mat4_from_rotation_translation(R, t):
    M = mat4_identity()
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def mat4_inverse_rt(R, t):
    # inverse of [R t; 0 1] with R orthonormal
    Rt = R.T
    ti = -Rt @ t
    M = mat4_identity()
    M[:3, :3] = Rt
    M[:3, 3] = ti
    return M

def openxr_projection_from_fov(fov: xr.Fovf, near=0.05, far=100.0):
    # Standard OpenXR projection matrix (right-handed, -Z forward in view space)
    # fov angles are in radians: left, right, up, down
    l = math.tan(fov.angle_left)
    r = math.tan(fov.angle_right)
    u = math.tan(fov.angle_up)
    d = math.tan(fov.angle_down)

    # OpenXR style (similar to Vulkan/GL with clip space)
    # We'll build an OpenGL-compatible matrix (NDC z in [-1,1])
    # Using common formula for asymmetric frustum.
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (r - l)
    M[1, 1] = 2.0 / (u - d)
    M[0, 2] = (r + l) / (r - l)
    M[1, 2] = (u + d) / (u - d)
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -(2.0 * far * near) / (far - near)
    M[3, 2] = -1.0
    return M


# ==========================================================
# Minimal OpenGL renderer (simple lit-ish color)
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

class SimpleRenderer:
    def __init__(self):
        vs_src = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;

        uniform mat4 uMVP;

        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        fs_src = r"""
        #version 330 core
        uniform vec4 uColor;
        out vec4 oColor;
        void main() {
            oColor = uColor;
        }
        """
        vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
        fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
        self.prog = link_program(vs, fs)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        self.uMVP = GL.glGetUniformLocation(self.prog, "uMVP")
        self.uColor = GL.glGetUniformLocation(self.prog, "uColor")

        self.vao_cube, self.vbo_cube, self.cube_count = self._make_unit_cube()
        self.vao_plane, self.vbo_plane, self.plane_count = self._make_unit_plane()

    def _make_unit_cube(self):
        # Unit cube centered at origin, size 1
        # 12 triangles => 36 verts
        v = np.array([
            # +Z
            -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,  0.5, 0.5, 0.5,
            -0.5,-0.5, 0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
            # -Z
            -0.5,-0.5,-0.5,  0.5, 0.5,-0.5,  0.5,-0.5,-0.5,
            -0.5,-0.5,-0.5, -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,
            # +X
             0.5,-0.5,-0.5,  0.5, 0.5, 0.5,  0.5,-0.5, 0.5,
             0.5,-0.5,-0.5,  0.5, 0.5,-0.5,  0.5, 0.5, 0.5,
            # -X
            -0.5,-0.5,-0.5, -0.5,-0.5, 0.5, -0.5, 0.5, 0.5,
            -0.5,-0.5,-0.5, -0.5, 0.5, 0.5, -0.5, 0.5,-0.5,
            # +Y
            -0.5, 0.5,-0.5, -0.5, 0.5, 0.5,  0.5, 0.5, 0.5,
            -0.5, 0.5,-0.5,  0.5, 0.5, 0.5,  0.5, 0.5,-0.5,
            # -Y
            -0.5,-0.5,-0.5,  0.5,-0.5, 0.5, -0.5,-0.5, 0.5,
            -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5,-0.5, 0.5,
        ], dtype=np.float32)

        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 12, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return vao, vbo, int(len(v)//3)

    def _make_unit_plane(self):
        # Unit quad in XY centered, z=0, two triangles
        v = np.array([
            -0.5,-0.5,0.0,  0.5,-0.5,0.0,  0.5, 0.5,0.0,
            -0.5,-0.5,0.0,  0.5, 0.5,0.0, -0.5, 0.5,0.0,
        ], dtype=np.float32)

        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 12, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return vao, vbo, int(len(v)//3)

    def draw_cube(self, mvp, color_rgba):
        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.uMVP, 1, True, mvp)  # True => row-major numpy
        GL.glUniform4f(self.uColor, *color_rgba)
        GL.glBindVertexArray(self.vao_cube)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.cube_count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def draw_plane(self, mvp, color_rgba):
        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.uMVP, 1, True, mvp)
        GL.glUniform4f(self.uColor, *color_rgba)
        GL.glBindVertexArray(self.vao_plane)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.plane_count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


# ==========================================================
# OpenXR runtime (main process)
# ==========================================================
class XrRuntime:
    def __init__(self, title="VR OpenGL Game-Style"):
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

        app_info = xr.ApplicationInfo("vr_opengl_game_style", 1, "pyopenxr", 0, xr.XR_CURRENT_API_VERSION)
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
        self.depth_rb = GL.glGenRenderbuffers(1)

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
        try: GL.glDeleteRenderbuffers(1, [self.depth_rb])
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
        def pr(name, val): print(f"[DBG] {name}: {val}")

        if last["grip"] is None or grip != last["grip"]: pr("Grip", int(grip))
        if last["a"] is None or a != last["a"]: pr("Button A", int(a))
        if last["b"] is None or b != last["b"]: pr("Button B", int(b))
        if last["trig"] is None or abs(trigger - last["trig"]) > 0.05: pr("Trigger", f"{trigger:.2f}")
        if last["sx"] is None or abs(sx - last["sx"]) > 0.20 or (abs(sx) < 0.05 and abs(last["sx"]) >= 0.05):
            pr("Thumbstick X", f"{sx:.2f}")
        if last["sy"] is None or abs(sy - last["sy"]) > 0.20 or (abs(sy) < 0.05 and abs(last["sy"]) >= 0.05):
            pr("Thumbstick Y", f"{sy:.2f}")
        if last["pose_valid"] is None or pose_valid != last["pose_valid"]: pr("Hand pose valid", int(pose_valid))

        last["grip"] = grip
        last["a"] = a
        last["b"] = b
        last["trig"] = trigger
        last["sx"] = sx
        last["sy"] = sy
        last["pose_valid"] = pose_valid


# ==========================================================
# "Simulation" state (replace with Genesis if needed)
# ==========================================================
class SimState:
    def __init__(self):
        self.active_idx = 0  # gray
        self.last_cube_pos = START_POS.copy()
        self.trigger = 0.0

    def choose_color(self, button_a, button_b, stick_active, trigger):
        desired = "gray"
        if button_a:
            desired = "red"
        elif button_b:
            desired = "blue"
        elif stick_active:
            desired = "yellow"
        elif trigger > 0.2:
            desired = "green"
        return COLOR_NAMES.index(desired)

    def update(self, grip, trigger, button_a, button_b, sx, sy, hand_pos):
        stick_active = (abs(sx) > STICK_DEADZONE) or (abs(sy) > STICK_DEADZONE)
        desired_idx = self.choose_color(button_a, button_b, stick_active, trigger)
        if desired_idx != self.active_idx:
            self.active_idx = desired_idx

        self.trigger = trigger

        if grip and hand_pos is not None:
            hx, hy, hz = hand_pos
            gp = map_pos_xr_to_gen((hx, hy, hz), world_t=WORLD_T)
            x = float(np.clip(gp[0], -0.45, 0.45))
            y = float(np.clip(gp[1], -0.45, 0.45))
            z = float(RING_Z + 0.12 + 0.25 * trigger)
            self.last_cube_pos[:] = (x, y, z)


# ==========================================================
# Render scene objects (plane + rails + active cube)
# ==========================================================
def draw_scene(renderer: SimpleRenderer, VP: np.ndarray, sim: SimState):
    # Background clear
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_LESS)

    # Ground plane (big)
    plane_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plane_scale = np.array([6.0, 6.0, 1.0], dtype=np.float32)
    M_plane = mat4_translate(plane_pos) @ mat4_scale(plane_scale[0], plane_scale[1], plane_scale[2])
    mvp_plane = VP @ M_plane
    renderer.draw_plane(mvp_plane, (0.25, 0.25, 0.25, 1.0))

    # Rails (4 boxes)
    rail_color = (0.35, 0.35, 0.35, 1.0)
    # Each rail is drawn as a scaled cube
    def draw_box(center, size_xyz):
        M = mat4_translate(np.array(center, dtype=np.float32)) @ mat4_scale(size_xyz[0], size_xyz[1], size_xyz[2])
        renderer.draw_cube(VP @ M, rail_color)

    ring_z = RING_Z
    ring_outer = RING_OUTER
    hole = HOLE
    plate_t = PLATE_T
    rail_w = RAIL_W

    draw_box((0.0, -(hole/2.0 + rail_w/2.0), ring_z), (ring_outer, rail_w, plate_t))
    draw_box((0.0, +(hole/2.0 + rail_w/2.0), ring_z), (ring_outer, rail_w, plate_t))
    draw_box((-(hole/2.0 + rail_w/2.0), 0.0, ring_z), (rail_w, hole, plate_t))
    draw_box((+(hole/2.0 + rail_w/2.0), 0.0, ring_z), (rail_w, hole, plate_t))

    # Active cube
    cube_color = COLORS[COLOR_NAMES[sim.active_idx]]
    cube_pos = sim.last_cube_pos
    M_cube = mat4_translate(cube_pos) @ mat4_scale(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    renderer.draw_cube(VP @ M_cube, cube_color)


# ==========================================================
# Main loop
# ==========================================================
def main():
    rt = XrRuntime("VR OpenGL Game-Style (no readback)")

    print("[Config]")
    print("  WORLD_T =", WORLD_T, "MAPPING =", MAPPING)
    print("  EYE_SEP_SCALE =", EYE_SEP_SCALE, "SWAP_EYES =", SWAP_EYES)
    print("  DEBUG_INPUTS =", DEBUG_INPUTS)

    renderer = SimpleRenderer()
    sim = SimState()

    sdl_event = sdl2.SDL_Event()
    running = True

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
                xr.poll_event(rt.instance)
            except xr.exception.EventUnavailable:
                break

        frame_state = xr.wait_frame(rt.session)
        xr.begin_frame(rt.session)

        # Inputs
        try:
            xr.sync_actions(rt.session, xr.ActionsSyncInfo(active_action_sets=[rt.active_action_set]))
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
            g = xr.get_action_state_boolean(rt.session, xr.ActionStateGetInfo(action=rt.grip_action, subaction_path=xr.NULL_PATH))
            t = xr.get_action_state_float(rt.session, xr.ActionStateGetInfo(action=rt.trigger_action, subaction_path=xr.NULL_PATH))
            a = xr.get_action_state_boolean(rt.session, xr.ActionStateGetInfo(action=rt.button_a_action, subaction_path=xr.NULL_PATH))
            b = xr.get_action_state_boolean(rt.session, xr.ActionStateGetInfo(action=rt.button_b_action, subaction_path=xr.NULL_PATH))
            ax = xr.get_action_state_float(rt.session, xr.ActionStateGetInfo(action=rt.thumbstick_x_action, subaction_path=xr.NULL_PATH))
            ay = xr.get_action_state_float(rt.session, xr.ActionStateGetInfo(action=rt.thumbstick_y_action, subaction_path=xr.NULL_PATH))

            grip = bool(g.current_state)
            trigger = float(t.current_state)
            button_a = bool(a.current_state)
            button_b = bool(b.current_state)
            sx, sy = float(ax.current_state), float(ay.current_state)

            loc = xr.locate_space(rt.hand_space, rt.reference_space, frame_state.predicted_display_time)
            flags = loc.location_flags
            pose_valid = bool((flags & xr.SpaceLocationFlags.POSITION_VALID_BIT) and (flags & xr.SpaceLocationFlags.ORIENTATION_VALID_BIT))
            if pose_valid:
                p = loc.pose.position
                hand_pos = (float(p.x), float(p.y), float(p.z))
        except Exception:
            pass

        rt._dbg_print_changes(grip, trigger, button_a, button_b, sx, sy, pose_valid)

        # Views
        try:
            _, located_views = xr.locate_views(
                rt.session,
                xr.ViewLocateInfo(
                    view_configuration_type=rt.view_config_type,
                    display_time=frame_state.predicted_display_time,
                    space=rt.reference_space
                ),
            )
        except Exception:
            xr.end_frame(rt.session, xr.FrameEndInfo(display_time=frame_state.predicted_display_time,
                                                     environment_blend_mode=rt.environment_blend_mode,
                                                     layers=[]))
            continue

        # Update "sim" (replace with Genesis physics update if desired)
        sim.update(grip, trigger, button_a, button_b, sx, sy, hand_pos)

        # Eye separation scaling (keep your old behavior)
        head = None
        p0 = located_views[0].pose.position
        p1 = located_views[1].pose.position
        head = 0.5 * np.array([p0.x + p1.x, p0.y + p1.y, p0.z + p1.z], dtype=np.float32)

        proj_views = []
        for eye_index, sc in enumerate(rt.swapchains):
            w, h = rt.image_sizes[eye_index]

            img_idx = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
            gl_image = rt.swapchain_images[eye_index][img_idx].image

            # Bind swapchain texture to FBO
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, rt.fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, gl_image, 0)

            # Depth buffer
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rt.depth_rb)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rt.depth_rb)

            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.05, 0.05, 0.06, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Build projection from OpenXR FOV
            P = openxr_projection_from_fov(located_views[eye_index].fov, near=0.05, far=100.0)

            # Build view from OpenXR pose
            p = located_views[eye_index].pose.position
            q = located_views[eye_index].pose.orientation

            # Apply your EYE_SEP_SCALE (optional, matches original code intent)
            pi = np.array([p.x, p.y, p.z], dtype=np.float32)
            pi = head + EYE_SEP_SCALE * (pi - head)

            R_xr = quat_to_rotmat_xyzw(q.x, q.y, q.z, q.w)
            t_xr = pi

            # Convert to your "genesis/world" mapping
            # We map basis vectors through map_vec_xr_to_gen, and position through map_pos_xr_to_gen
            right = map_vec_xr_to_gen(R_xr[:, 0])
            up    = map_vec_xr_to_gen(R_xr[:, 1])
            fwd   = map_vec_xr_to_gen(-R_xr[:, 2])  # forward

            # Re-orthonormalize to be safe
            fwd = normalize(fwd)
            right = normalize(np.cross(up, fwd))
            up = normalize(np.cross(fwd, right))

            Rg = np.stack([right, up, fwd], axis=1).astype(np.float32)
            tg = map_pos_xr_to_gen((t_xr[0], t_xr[1], t_xr[2]), world_t=WORLD_T)

            V = mat4_inverse_rt(Rg, tg)

            VP = P @ V

            # Draw world
            draw_scene(renderer, VP, sim)

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

        if SWAP_EYES:
            proj_views[0], proj_views[1] = proj_views[1], proj_views[0]

        proj_layer = xr.CompositionLayerProjection(
            layer_flags=xr.CompositionLayerFlags.NONE,
            space=rt.reference_space,
            views=proj_views,
        )

        xr.end_frame(
            rt.session,
            xr.FrameEndInfo(display_time=frame_state.predicted_display_time,
                            environment_blend_mode=rt.environment_blend_mode,
                            layers=[ctypes.pointer(proj_layer)]),
        )

    rt.shutdown()


if __name__ == "__main__":
    main()
