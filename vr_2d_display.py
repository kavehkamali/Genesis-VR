# vr_genesis_bridge_full.py
#
# Fixes + features requested:
#   1) Controllers/input keep working after removing/putting headset back on
#      - Properly handles OpenXR session lifecycle via SESSION_STATE_CHANGED events
#      - Only calls xrWaitFrame/xrBeginFrame while session is RUNNING
#      - Re-begins session when runtime returns to READY (after STOPPING)
#      - Gates input on FOCUSED state (recommended for Quest Link)
#
#   2) 2D "Genesis visualization" inside VR while staying focused
#      - Renders a spectator camera view (your proxy scene) to an offscreen texture (FBO)
#      - Displays that texture on a floating quad in front of the HMD
#
#   3) Keeps the "black headset after shadow FBO" fix
#      - Restores glDrawBuffer/glReadBuffer to COLOR_ATTACHMENT0 on swapchain FBO
#
# Notes:
#   - This shows a 2D view of the SAME proxy scene you're already rendering (plane/cube/franka links).
#     It does NOT use Genesis' own GUI viewer window.
#
# Windows-only (WGL + SDL). Requires: pyopenxr (import xr), PyOpenGL, pysdl2, genesis.

import ctypes
import os
import platform
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

import genesis as gs


# ==========================================================
# Config
# ==========================================================
def f32(x):
    return np.asarray(x, dtype=np.float32)


@dataclass(frozen=True)
class AppConfig:
    title: str = "VR + Genesis Bridge (Physics in Genesis, Render in VR)"

    # XR scene offset in reference space (+X right, +Y up, -Z forward)
    scene_offset: np.ndarray = field(default_factory=lambda: f32([0.0, -1.2, -1.2]))

    # Input
    stick_deadzone: float = 0.30
    debug_inputs: bool = True
    swap_eyes: bool = False

    # GL / shadows
    shadow_map_size: int = 2048
    light_dir: np.ndarray = field(default_factory=lambda: f32([0.35, -1.0, 0.25]))

    # Clip
    near: float = 0.05
    far: float = 50.0

    # Genesis sim
    genesis_dt: float = 0.01
    genesis_substeps_per_vr_frame: int = 1
    genesis_backend: str = "gpu"  # "gpu" or "cpu"

    # Robot proxy draw
    robot_link_box_size: float = 0.03

    # 2D spectator view (render-to-texture)
    spectator_tex_size: int = 1024
    screen_distance_m: float = 0.85   # distance in front of head
    screen_width_m: float = 1.25      # physical width in meters
    screen_height_m: float = 0.75     # physical height in meters
    screen_down_m: float = 0.12       # place slightly below eye
    show_screen: bool = True          # can be toggled easily later

    # Palette
    color_names: Tuple[str, ...] = ("gray", "red", "blue", "green", "yellow")
    colors: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "gray": (0.6, 0.6, 0.6, 1.0),
            "red": (1.0, 0.1, 0.1, 1.0),
            "blue": (0.1, 0.3, 1.0, 1.0),
            "green": (0.2, 1.0, 0.2, 1.0),
            "yellow": (1.0, 1.0, 0.2, 1.0),
        }
    )


# ==========================================================
# Math utilities
# ==========================================================
def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def mat4_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def mat4_translate(t: np.ndarray) -> np.ndarray:
    M = mat4_identity()
    M[:3, 3] = t[:3]
    return M


def mat4_scale(sx: float, sy: float, sz: float) -> np.ndarray:
    M = mat4_identity()
    M[0, 0] = float(sx)
    M[1, 1] = float(sy)
    M[2, 2] = float(sz)
    return M


def mat4_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = mat4_identity()
    M[:3, :3] = R
    M[:3, 3] = t[:3]
    return M


def mat4_inverse_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = R.T
    ti = -Rt @ t
    M = mat4_identity()
    M[:3, :3] = Rt
    M[:3, 3] = ti
    return M


def openxr_projection_from_fov(fov: xr.Fovf, near: float, far: float) -> np.ndarray:
    l = math.tan(fov.angle_left)
    r = math.tan(fov.angle_right)
    u = math.tan(fov.angle_up)
    d = math.tan(fov.angle_down)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (r - l)
    M[1, 1] = 2.0 / (u - d)
    M[0, 2] = (r + l) / (r - l)
    M[1, 2] = (u + d) / (u - d)
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -(2.0 * far * near) / (far - near)
    M[3, 2] = -1.0
    return M


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = mat4_identity()
    M[0, 0:3] = s
    M[1, 0:3] = u
    M[2, 0:3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def perspective(fovy_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy_rad * 0.5)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


# ==========================================================
# GL helpers
# ==========================================================
def compile_shader(src: str, shader_type) -> int:
    sid = GL.glCreateShader(shader_type)
    GL.glShaderSource(sid, src)
    GL.glCompileShader(sid)
    ok = GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS)
    if not ok:
        raise RuntimeError(GL.glGetShaderInfoLog(sid).decode("utf-8", "ignore"))
    return sid


def link_program(vs: int, fs: int) -> int:
    pid = GL.glCreateProgram()
    GL.glAttachShader(pid, vs)
    GL.glAttachShader(pid, fs)
    GL.glLinkProgram(pid)
    ok = GL.glGetProgramiv(pid, GL.GL_LINK_STATUS)
    if not ok:
        raise RuntimeError(GL.glGetProgramInfoLog(pid).decode("utf-8", "ignore"))
    return pid


# ==========================================================
# Draw structures
# ==========================================================
@dataclass
class MeshHandle:
    vao: int
    count: int


@dataclass
class DrawItem:
    mesh: MeshHandle
    M: np.ndarray
    color: Tuple[float, float, float, float]


# ==========================================================
# Offscreen spectator (render-to-texture) target
# ==========================================================
class Spectator2D:
    def __init__(self, size: int = 1024):
        self.w = int(size)
        self.h = int(size)

        self.fbo = GL.glGenFramebuffers(1)
        self.tex = GL.glGenTextures(1)
        self.depth = GL.glGenRenderbuffers(1)

        # Color texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.w, self.h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # Depth renderbuffer
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, self.w, self.h)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)

        # FBO attach
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex, 0)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.depth)

        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Spectator FBO incomplete: {hex(int(status))}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def bind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.w, self.h)
        # Safety: make sure buffers are correct (some other pass might set NONE)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

    def unbind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)


def make_spectator_vp(cfg: AppConfig) -> np.ndarray:
    # A stable 2D-ish camera for your scene (feel free to tweak)
    eye = cfg.scene_offset + np.array([1.35, 0.55, 1.35], dtype=np.float32)
    target = cfg.scene_offset + np.array([0.0, 0.05, 0.25], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    V = look_at(eye, target, up)
    P = perspective(math.radians(55.0), aspect=1.0, near=cfg.near, far=cfg.far)
    return P @ V


# ==========================================================
# Renderer (textured + shadows + screen quad)
# ==========================================================
class Renderer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.light_dir = normalize(cfg.light_dir.astype(np.float32))
        self._init_programs()
        self._init_meshes()
        self._init_checker_texture()
        self._init_shadow_map()

    def _init_programs(self) -> None:
        # Main + shadow programs (same as your working version)
        vs_src = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec3 aNrm;
        layout(location=2) in vec2 aUV;

        uniform mat4 uM;
        uniform mat4 uVP;
        uniform mat4 uLightVP;

        out vec3 vNrm;
        out vec2 vUV;
        out vec4 vShadowCoord;

        void main() {
            vec4 wp = uM * vec4(aPos, 1.0);
            vNrm = mat3(uM) * aNrm;
            vUV = aUV;
            vShadowCoord = uLightVP * wp;
            gl_Position = uVP * wp;
        }
        """

        fs_src = r"""
        #version 330 core
        in vec3 vNrm;
        in vec2 vUV;
        in vec4 vShadowCoord;

        uniform vec4 uColor;
        uniform vec3 uLightDir;
        uniform sampler2D uTex;
        uniform sampler2DShadow uShadowMap;

        out vec4 oColor;

        float shadow_factor(vec4 sc) {
            vec3 proj = sc.xyz / sc.w;
            vec3 uvz = proj * 0.5 + 0.5;
            if (uvz.x < 0.0 || uvz.x > 1.0 || uvz.y < 0.0 || uvz.y > 1.0) return 1.0;
            float bias = 0.0025;
            return texture(uShadowMap, vec3(uvz.xy, uvz.z - bias));
        }

        void main() {
            vec3 n = normalize(vNrm);
            float ndl = max(dot(n, -uLightDir), 0.0);

            vec3 tex = texture(uTex, vUV).rgb;
            vec3 base = tex * uColor.rgb;

            float sh = shadow_factor(vShadowCoord);
            float ambient = 0.25;
            float diffuse = 0.85 * ndl;

            vec3 lit = base * (ambient + diffuse * sh);
            oColor = vec4(lit, uColor.a);
        }
        """

        shadow_vs = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uM;
        uniform mat4 uLightVP;
        void main() { gl_Position = uLightVP * (uM * vec4(aPos, 1.0)); }
        """

        shadow_fs = r"""
        #version 330 core
        void main() { }
        """

        vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
        fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
        self.prog = link_program(vs, fs)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        svs = compile_shader(shadow_vs, GL.GL_VERTEX_SHADER)
        sfs = compile_shader(shadow_fs, GL.GL_FRAGMENT_SHADER)
        self.shadow_prog = link_program(svs, sfs)
        GL.glDeleteShader(svs)
        GL.glDeleteShader(sfs)

        self.uM = GL.glGetUniformLocation(self.prog, "uM")
        self.uVP = GL.glGetUniformLocation(self.prog, "uVP")
        self.uLightVP = GL.glGetUniformLocation(self.prog, "uLightVP")
        self.uColor = GL.glGetUniformLocation(self.prog, "uColor")
        self.uLightDir = GL.glGetUniformLocation(self.prog, "uLightDir")
        self.uTex = GL.glGetUniformLocation(self.prog, "uTex")
        self.uShadowMap = GL.glGetUniformLocation(self.prog, "uShadowMap")

        self.s_uM = GL.glGetUniformLocation(self.shadow_prog, "uM")
        self.s_uLightVP = GL.glGetUniformLocation(self.shadow_prog, "uLightVP")

        # Screen quad program (unlit texture)
        screen_vs = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=2) in vec2 aUV;

        uniform mat4 uM;
        uniform mat4 uVP;

        out vec2 vUV;

        void main() {
            vUV = aUV;
            gl_Position = uVP * (uM * vec4(aPos, 1.0));
        }
        """
        screen_fs = r"""
        #version 330 core
        in vec2 vUV;
        uniform sampler2D uScreenTex;
        out vec4 oColor;
        void main() {
            oColor = texture(uScreenTex, vUV);
        }
        """

        pvs = compile_shader(screen_vs, GL.GL_VERTEX_SHADER)
        pfs = compile_shader(screen_fs, GL.GL_FRAGMENT_SHADER)
        self.screen_prog = link_program(pvs, pfs)
        GL.glDeleteShader(pvs)
        GL.glDeleteShader(pfs)

        self.screen_uM = GL.glGetUniformLocation(self.screen_prog, "uM")
        self.screen_uVP = GL.glGetUniformLocation(self.screen_prog, "uVP")
        self.screen_uTex = GL.glGetUniformLocation(self.screen_prog, "uScreenTex")

    def _init_meshes(self) -> None:
        self.mesh_cube = self._make_cube()
        self.mesh_plane = self._make_plane()

    def _make_cube(self) -> MeshHandle:
        data = []

        def add_face(n, verts, uvs):
            for (p, uv) in zip(verts, uvs):
                data.extend([p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]])

        add_face(
            (0, 0, 1),
            [
                (-0.5, -0.5, 0.5),
                (0.5, -0.5, 0.5),
                (0.5, 0.5, 0.5),
                (-0.5, -0.5, 0.5),
                (0.5, 0.5, 0.5),
                (-0.5, 0.5, 0.5),
            ],
            [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)],
        )
        add_face(
            (0, 0, -1),
            [
                (-0.5, -0.5, -0.5),
                (0.5, 0.5, -0.5),
                (0.5, -0.5, -0.5),
                (-0.5, -0.5, -0.5),
                (-0.5, 0.5, -0.5),
                (0.5, 0.5, -0.5),
            ],
            [(0, 0), (1, 1), (1, 0), (0, 0), (0, 1), (1, 1)],
        )
        add_face(
            (1, 0, 0),
            [
                (0.5, -0.5, -0.5),
                (0.5, 0.5, 0.5),
                (0.5, -0.5, 0.5),
                (0.5, -0.5, -0.5),
                (0.5, 0.5, -0.5),
                (0.5, 0.5, 0.5),
            ],
            [(0, 0), (1, 1), (1, 0), (0, 0), (0, 1), (1, 1)],
        )
        add_face(
            (-1, 0, 0),
            [
                (-0.5, -0.5, -0.5),
                (-0.5, -0.5, 0.5),
                (-0.5, 0.5, 0.5),
                (-0.5, -0.5, -0.5),
                (-0.5, 0.5, 0.5),
                (-0.5, 0.5, -0.5),
            ],
            [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)],
        )
        add_face(
            (0, 1, 0),
            [
                (-0.5, 0.5, -0.5),
                (-0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                (-0.5, 0.5, -0.5),
                (0.5, 0.5, 0.5),
                (0.5, 0.5, -0.5),
            ],
            [(0, 0), (0, 1), (1, 1), (0, 0), (1, 1), (1, 0)],
        )
        add_face(
            (0, -1, 0),
            [
                (-0.5, -0.5, -0.5),
                (0.5, -0.5, 0.5),
                (-0.5, -0.5, 0.5),
                (-0.5, -0.5, -0.5),
                (0.5, -0.5, -0.5),
                (0.5, -0.5, 0.5),
            ],
            [(0, 0), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1)],
        )

        v = np.asarray(data, dtype=np.float32)
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)

        stride = 8 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(24))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return MeshHandle(vao=vao, count=int(len(v) / 8))

    def _make_plane(self) -> MeshHandle:
        # Plane in XZ, centered at origin, y=0. UVs 0..4 (we'll still sample fine).
        v = np.array(
            [
                -0.5, 0.0, -0.5,  0, 1, 0,  0, 0,
                 0.5, 0.0, -0.5,  0, 1, 0,  4, 0,
                 0.5, 0.0,  0.5,  0, 1, 0,  4, 4,

                -0.5, 0.0, -0.5,  0, 1, 0,  0, 0,
                 0.5, 0.0,  0.5,  0, 1, 0,  4, 4,
                -0.5, 0.0,  0.5,  0, 1, 0,  0, 4,
            ],
            dtype=np.float32,
        )

        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)

        stride = 8 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(24))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return MeshHandle(vao=vao, count=int(len(v) / 8))

    def _init_checker_texture(self) -> None:
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)

        w = h = 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                c = 40 if ((x // 16) ^ (y // 16)) & 1 else 200
                img[y, x] = (c, c, c)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self.checker_tex = tex

    def _init_shadow_map(self) -> None:
        size = int(self.cfg.shadow_map_size)
        self.shadow_fbo = GL.glGenFramebuffers(1)
        self.shadow_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT24,
            size,
            size,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_UNSIGNED_INT,
            None,
        )

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_COMPARE_REF_TO_TEXTURE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LEQUAL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.shadow_tex, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Shadow FBO incomplete: {status}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def draw_shadow_pass(self, light_vp: np.ndarray, items: List[DrawItem]) -> None:
        size = int(self.cfg.shadow_map_size)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_fbo)
        GL.glViewport(0, 0, size, size)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glCullFace(GL.GL_FRONT)
        GL.glEnable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.shadow_prog)
        GL.glUniformMatrix4fv(self.s_uLightVP, 1, True, light_vp)

        for it in items:
            GL.glUniformMatrix4fv(self.s_uM, 1, True, it.M)
            GL.glBindVertexArray(it.mesh.vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, it.mesh.count)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def draw_main_pass(self, vp: np.ndarray, light_vp: np.ndarray, items: List[DrawItem]) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glCullFace(GL.GL_BACK)
        GL.glEnable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.uVP, 1, True, vp)
        GL.glUniformMatrix4fv(self.uLightVP, 1, True, light_vp)
        GL.glUniform3f(self.uLightDir, float(self.light_dir[0]), float(self.light_dir[1]), float(self.light_dir[2]))

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.checker_tex)
        GL.glUniform1i(self.uTex, 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_tex)
        GL.glUniform1i(self.uShadowMap, 1)

        for it in items:
            GL.glUniformMatrix4fv(self.uM, 1, True, it.M)
            c = it.color
            GL.glUniform4f(self.uColor, float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            GL.glBindVertexArray(it.mesh.vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, it.mesh.count)

        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_CULL_FACE)

    def draw_screen_quad(self, vp: np.ndarray, M: np.ndarray, tex_id: int) -> None:
        # Draw a textured quad (uses mesh_plane)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        GL.glUseProgram(self.screen_prog)
        GL.glUniformMatrix4fv(self.screen_uVP, 1, True, vp)
        GL.glUniformMatrix4fv(self.screen_uM, 1, True, M)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glUniform1i(self.screen_uTex, 0)

        GL.glBindVertexArray(self.mesh_plane.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.mesh_plane.count)

        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


# ==========================================================
# XR runtime
# ==========================================================
@dataclass
class InputState:
    grip: bool = False
    trigger: float = 0.0
    button_a: bool = False
    button_b: bool = False
    sx: float = 0.0
    sy: float = 0.0
    hand_pos_world: Optional[Tuple[float, float, float]] = None
    hand_pose_valid: bool = False


class XrRuntime:
    def __init__(self, cfg: AppConfig):
        if platform.system() != "Windows":
            raise RuntimeError("Windows-only (WGL + SDL).")

        self.cfg = cfg

        # Session lifecycle flags (this fixes your headset-remove/reput bug)
        self.session_state = xr.SessionState.IDLE
        self.session_running = False
        self.exit_requested = False

        self._init_xr()
        self._init_window_gl()
        self._init_session()
        self._init_spaces()
        self._init_swapchains()
        self._init_fbo_depth()
        self._init_actions()

        self.environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE

        self._dbg_last = {"grip": None, "a": None, "b": None, "trig": None, "sx": None, "sy": None, "pose_valid": None}

        # Store the context that OpenXR session was bound to
        self.hdc = WGL.wglGetCurrentDC()
        self.hglrc = WGL.wglGetCurrentContext()

        # Poll initial events once; many runtimes will quickly go to READY
        self.poll_xr_events()

    def _init_xr(self) -> None:
        exts = xr.enumerate_instance_extension_properties()
        avail = set(
            (e.extension_name.decode("utf-8", "ignore") if isinstance(e.extension_name, bytes) else e.extension_name)
            for e in exts
        )
        req = xr.KHR_OPENGL_ENABLE_EXTENSION_NAME
        req = req.decode("utf-8", "ignore") if isinstance(req, bytes) else req
        if req not in avail:
            raise RuntimeError("XR_KHR_opengl_enable not available")

        app_info = xr.ApplicationInfo("vr_genesis_bridge", 1, "pyopenxr", 0, xr.XR_CURRENT_API_VERSION)
        self.instance = xr.create_instance(xr.InstanceCreateInfo(application_info=app_info, enabled_extension_names=[req]))

        self.system_id = xr.get_system(self.instance, xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self.view_config_views = xr.enumerate_view_configuration_views(self.instance, self.system_id, self.view_config_type)

    def _init_window_gl(self) -> None:
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError("SDL_Init failed")

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 0)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.SDL_GL_CONTEXT_PROFILE_CORE)

        ww = self.view_config_views[0].recommended_image_rect_width // 2
        wh = self.view_config_views[0].recommended_image_rect_height // 2

        self.window = sdl2.SDL_CreateWindow(
            self.cfg.title.encode("utf-8"),
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            ww,
            wh,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_SHOWN,
        )
        if not self.window:
            raise RuntimeError("Failed to create SDL window")

        self.gl_context = sdl2.SDL_GL_CreateContext(self.window)
        if not self.gl_context:
            raise RuntimeError("Failed to create OpenGL context")

        sdl2.SDL_GL_MakeCurrent(self.window, self.gl_context)
        sdl2.SDL_GL_SetSwapInterval(0)

    def _init_session(self) -> None:
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

    def _init_spaces(self) -> None:
        identity = xr.Posef(orientation=xr.Quaternionf(0, 0, 0, 1), position=xr.Vector3f(0, 0, 0))
        try:
            self.reference_space = xr.create_reference_space(
                self.session,
                xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.STAGE, pose_in_reference_space=identity),
            )
        except Exception:
            self.reference_space = xr.create_reference_space(
                self.session,
                xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.LOCAL, pose_in_reference_space=identity),
            )

    def _init_swapchains(self) -> None:
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
                    width=w,
                    height=h,
                    face_count=1,
                    array_size=1,
                    mip_count=1,
                ),
            )
            imgs = xr.enumerate_swapchain_images(sc, xr.SwapchainImageOpenGLKHR)
            self.swapchains.append(sc)
            self.swapchain_images.append(imgs)
            self.image_sizes.append((w, h))

    def _init_fbo_depth(self) -> None:
        self.fbo = GL.glGenFramebuffers(1)
        self.depth_rb = GL.glGenRenderbuffers(1)

    def _init_actions(self) -> None:
        self.action_set = xr.create_action_set(
            self.instance,
            xr.ActionSetCreateInfo(action_set_name="teleop", localized_action_set_name="Teleop", priority=0),
        )
        self.right_hand_path = xr.string_to_path(self.instance, "/user/hand/right")

        self.grip_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="right_grip",
                action_type=xr.ActionType.BOOLEAN_INPUT,
                localized_action_name="Right Grip",
                subaction_paths=[self.right_hand_path],
            ),
        )
        self.trigger_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="right_trigger",
                action_type=xr.ActionType.FLOAT_INPUT,
                localized_action_name="Right Trigger",
                subaction_paths=[self.right_hand_path],
            ),
        )
        self.hand_pose_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="right_hand_pose",
                action_type=xr.ActionType.POSE_INPUT,
                localized_action_name="Right Hand Pose",
                subaction_paths=[self.right_hand_path],
            ),
        )
        self.button_a_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="button_a",
                action_type=xr.ActionType.BOOLEAN_INPUT,
                localized_action_name="Button A",
                subaction_paths=[self.right_hand_path],
            ),
        )
        self.button_b_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="button_b",
                action_type=xr.ActionType.BOOLEAN_INPUT,
                localized_action_name="Button B",
                subaction_paths=[self.right_hand_path],
            ),
        )
        self.thumbstick_x_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="thumbstick_x",
                action_type=xr.ActionType.FLOAT_INPUT,
                localized_action_name="Thumbstick X",
                subaction_paths=[self.right_hand_path],
            ),
        )
        self.thumbstick_y_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="thumbstick_y",
                action_type=xr.ActionType.FLOAT_INPUT,
                localized_action_name="Thumbstick Y",
                subaction_paths=[self.right_hand_path],
            ),
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

        identity = xr.Posef(orientation=xr.Quaternionf(0, 0, 0, 1), position=xr.Vector3f(0, 0, 0))
        self.hand_space = xr.create_action_space(
            self.session,
            xr.ActionSpaceCreateInfo(
                action=self.hand_pose_action,
                subaction_path=self.right_hand_path,
                pose_in_action_space=identity,
            ),
        )

        self.active_action_set = xr.ActiveActionSet(action_set=self.action_set, subaction_path=xr.NULL_PATH)

    def pump_sdl_events(self) -> bool:
        sdl_event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(ctypes.byref(sdl_event)) != 0:
            if sdl_event.type == sdl2.SDL_QUIT:
                return False
            if sdl_event.type == sdl2.SDL_KEYDOWN and sdl_event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                return False
        return True

    def _extract_session_state(self, ev) -> Optional[xr.SessionState]:
        # Tries multiple pyopenxr representations
        try:
            # Some builds return an EventDataSessionStateChanged directly
            if hasattr(ev, "state"):
                return ev.state
        except Exception:
            pass

        # EventDataBuffer casting attempt
        try:
            # If ev is EventDataBuffer-like, it should have a "type" field and be ctypes-backed
            if hasattr(ev, "type") and int(ev.type) == int(xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED):
                try:
                    s = xr.EventDataSessionStateChanged.from_buffer(ev)  # if available
                    return s.state
                except Exception:
                    try:
                        s = ctypes.cast(ctypes.pointer(ev), ctypes.POINTER(xr.EventDataSessionStateChanged)).contents
                        return s.state
                    except Exception:
                        return None
        except Exception:
            return None

        return None

    def poll_xr_events(self) -> None:
        # Proper session lifecycle handling (fixes "remove headset -> input dead until restart")
        while True:
            try:
                ev = xr.poll_event(self.instance)
            except xr.exception.EventUnavailable:
                break
            except Exception:
                break

            st = self._extract_session_state(ev)
            if st is not None:
                self.session_state = st

                if self.cfg.debug_inputs:
                    print("[XR] Session state =", self.session_state)

                # Begin session when READY
                if self.session_state == xr.SessionState.READY and not self.session_running:
                    try:
                        xr.begin_session(self.session, xr.SessionBeginInfo(primary_view_configuration_type=self.view_config_type))
                        self.session_running = True
                        if self.cfg.debug_inputs:
                            print("[XR] xrBeginSession()")
                    except Exception as e:
                        print("[XR][WARN] xrBeginSession failed:", repr(e))

                # End session when STOPPING
                elif self.session_state == xr.SessionState.STOPPING and self.session_running:
                    try:
                        xr.end_session(self.session)
                        self.session_running = False
                        if self.cfg.debug_inputs:
                            print("[XR] xrEndSession()")
                    except Exception as e:
                        print("[XR][WARN] xrEndSession failed:", repr(e))

                # Exit or loss pending -> quit loop
                elif self.session_state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                    self.exit_requested = True

            # Some runtimes can also request quit via instance-loss pending events,
            # but we keep it simple here.

    def wait_begin_frame(self) -> xr.FrameState:
        frame_state = xr.wait_frame(self.session)
        xr.begin_frame(self.session)
        return frame_state

    def locate_views(self, predicted_time) -> Optional[List[xr.View]]:
        try:
            _, located_views = xr.locate_views(
                self.session,
                xr.ViewLocateInfo(
                    view_configuration_type=self.view_config_type,
                    display_time=predicted_time,
                    space=self.reference_space,
                ),
            )
            return located_views
        except Exception:
            return None

    def sync_actions(self) -> None:
        try:
            xr.sync_actions(self.session, xr.ActionsSyncInfo(active_action_sets=[self.active_action_set]))
        except Exception as e:
            if self.cfg.debug_inputs:
                print("[XR][WARN] sync_actions failed:", repr(e))

    def poll_inputs(self, predicted_time) -> InputState:
        inp = InputState()
        try:
            g = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(action=self.grip_action, subaction_path=xr.NULL_PATH))
            t = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(action=self.trigger_action, subaction_path=xr.NULL_PATH))
            a = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(action=self.button_a_action, subaction_path=xr.NULL_PATH))
            b = xr.get_action_state_boolean(self.session, xr.ActionStateGetInfo(action=self.button_b_action, subaction_path=xr.NULL_PATH))
            ax = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(action=self.thumbstick_x_action, subaction_path=xr.NULL_PATH))
            ay = xr.get_action_state_float(self.session, xr.ActionStateGetInfo(action=self.thumbstick_y_action, subaction_path=xr.NULL_PATH))

            inp.grip = bool(g.current_state)
            inp.trigger = float(t.current_state)
            inp.button_a = bool(a.current_state)
            inp.button_b = bool(b.current_state)
            inp.sx, inp.sy = float(ax.current_state), float(ay.current_state)

            loc = xr.locate_space(self.hand_space, self.reference_space, predicted_time)
            flags = loc.location_flags
            inp.hand_pose_valid = bool((flags & xr.SpaceLocationFlags.POSITION_VALID_BIT) and (flags & xr.SpaceLocationFlags.ORIENTATION_VALID_BIT))
            if inp.hand_pose_valid:
                p = loc.pose.position
                inp.hand_pos_world = (float(p.x), float(p.y), float(p.z))
        except Exception as e:
            if self.cfg.debug_inputs:
                print("[XR][WARN] poll_inputs failed:", repr(e))

        self._dbg_inputs(inp)
        return inp

    def _dbg_inputs(self, inp: InputState) -> None:
        if not self.cfg.debug_inputs:
            return
        last = self._dbg_last

        def pr(name, val):
            print(f"[DBG] {name}: {val}")

        if last["grip"] is None or inp.grip != last["grip"]:
            pr("Grip", int(inp.grip))
        if last["a"] is None or inp.button_a != last["a"]:
            pr("Button A", int(inp.button_a))
        if last["b"] is None or inp.button_b != last["b"]:
            pr("Button B", int(inp.button_b))
        if last["trig"] is None or abs(inp.trigger - last["trig"]) > 0.05:
            pr("Trigger", f"{inp.trigger:.2f}")
        if last["sx"] is None or abs(inp.sx - last["sx"]) > 0.20:
            pr("Thumbstick X", f"{inp.sx:.2f}")
        if last["sy"] is None or abs(inp.sy - last["sy"]) > 0.20:
            pr("Thumbstick Y", f"{inp.sy:.2f}")
        if last["pose_valid"] is None or inp.hand_pose_valid != last["pose_valid"]:
            pr("Hand pose valid", int(inp.hand_pose_valid))

        last["grip"] = inp.grip
        last["a"] = inp.button_a
        last["b"] = inp.button_b
        last["trig"] = inp.trigger
        last["sx"] = inp.sx
        last["sy"] = inp.sy
        last["pose_valid"] = inp.hand_pose_valid

    def end_frame_empty(self, predicted_time) -> None:
        try:
            xr.end_frame(
                self.session,
                xr.FrameEndInfo(display_time=predicted_time, environment_blend_mode=self.environment_blend_mode, layers=[]),
            )
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            if self.session_running:
                xr.end_session(self.session)
        except Exception:
            pass
        try:
            xr.destroy_session(self.session)
        except Exception:
            pass
        try:
            xr.destroy_instance(self.instance)
        except Exception:
            pass
        try:
            GL.glDeleteFramebuffers(1, [self.fbo])
        except Exception:
            pass
        try:
            GL.glDeleteRenderbuffers(1, [self.depth_rb])
        except Exception:
            pass
        try:
            sdl2.SDL_GL_DeleteContext(self.gl_context)
        except Exception:
            pass
        try:
            sdl2.SDL_DestroyWindow(self.window)
        except Exception:
            pass
        try:
            sdl2.SDL_Quit()
        except Exception:
            pass


# ==========================================================
# Genesis scene builder
# ==========================================================
@dataclass
class GenesisHandles:
    scene: "gs.Scene"
    entities: Dict[str, object]


def build_genesis_scene(cfg: AppConfig) -> GenesisHandles:
    backend = gs.gpu if cfg.genesis_backend.lower() == "gpu" else gs.cpu
    gs.init(backend=backend)

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=cfg.genesis_dt), show_viewer=False)

    entities: Dict[str, object] = {}
    entities["plane"] = scene.add_entity(gs.morphs.Plane())
    entities["cube"] = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
    entities["franka"] = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    scene.build()
    return GenesisHandles(scene=scene, entities=entities)


# ==========================================================
# Genesis → Render proxy extraction
# ==========================================================
class GenesisProxyExtractor:
    def __init__(self, cfg: AppConfig, renderer: Renderer, handles: GenesisHandles):
        self.cfg = cfg
        self.r = renderer
        self.h = handles

    def _try_get_pos_quat(self, ent) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            if hasattr(ent, "get_pos") and hasattr(ent, "get_quat"):
                p = np.asarray(ent.get_pos()).reshape(-1).astype(np.float32, copy=False)
                q = np.asarray(ent.get_quat()).reshape(-1).astype(np.float32, copy=False)
                if p.size >= 3 and q.size >= 4:
                    return p[:3], q[:4]
        except Exception:
            pass
        return None

    def _world_from_genesis(self, p: np.ndarray) -> np.ndarray:
        return self.cfg.scene_offset + p

    def compute_light_vp(self) -> np.ndarray:
        center = self.cfg.scene_offset + np.array([0.0, 0.0, 0.4], dtype=np.float32)
        light_pos = center - self.r.light_dir * 4.0
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(up, self.r.light_dir)) > 0.95:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        V = look_at(light_pos, center, up)

        l, r = -2.5, 2.5
        b, t = -2.5, 2.5
        n, f = 0.1, 10.0

        P = np.array(
            [
                [2 / (r - l), 0, 0, -(r + l) / (r - l)],
                [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                [0, 0, -2 / (f - n), -(f + n) / (f - n)],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return P @ V

    def build_draw_items(self) -> Tuple[List[DrawItem], List[DrawItem], np.ndarray]:
        main: List[DrawItem] = []
        shadow: List[DrawItem] = []

        light_vp = self.compute_light_vp()

        # Ground
        plane_center = self.cfg.scene_offset + np.array([0.0, -0.05, 0.0], dtype=np.float32)
        M_plane = mat4_translate(plane_center) @ mat4_scale(8.0, 1.0, 8.0)
        main.append(DrawItem(self.r.mesh_plane, M_plane, (0.85, 0.85, 0.85, 1.0)))
        shadow.append(DrawItem(self.r.mesh_plane, M_plane, (0, 0, 0, 0)))

        # Cube
        cube = self.h.entities.get("cube")
        if cube is not None:
            pq = self._try_get_pos_quat(cube)
            if pq is not None:
                p, q = pq
                pw = self._world_from_genesis(p)
                R = quat_xyzw_to_rotmat(q)
                M = mat4_from_Rt(R, pw) @ mat4_scale(0.04, 0.04, 0.04)
                main.append(DrawItem(self.r.mesh_cube, M, (0.9, 0.3, 0.3, 1.0)))
                shadow.append(DrawItem(self.r.mesh_cube, M, (0, 0, 0, 0)))

        # Franka as link proxy cubes
        franka = self.h.entities.get("franka")
        if franka is not None:
            try:
                if hasattr(franka, "get_links_pos") and hasattr(franka, "get_links_quat"):
                    lp = np.asarray(franka.get_links_pos()).astype(np.float32, copy=False)
                    lq = np.asarray(franka.get_links_quat()).astype(np.float32, copy=False)
                    n = min(lp.shape[0], lq.shape[0])
                    s = float(self.cfg.robot_link_box_size)
                    for i in range(n):
                        pw = self._world_from_genesis(lp[i])
                        R = quat_xyzw_to_rotmat(lq[i])
                        M = mat4_from_Rt(R, pw) @ mat4_scale(s, s, s)
                        main.append(DrawItem(self.r.mesh_cube, M, (0.45, 0.55, 0.95, 1.0)))
                        shadow.append(DrawItem(self.r.mesh_cube, M, (0, 0, 0, 0)))
            except Exception:
                pq = self._try_get_pos_quat(franka)
                if pq is not None:
                    p, q = pq
                    pw = self._world_from_genesis(p)
                    R = quat_xyzw_to_rotmat(q)
                    s = float(self.cfg.robot_link_box_size) * 2.5
                    M = mat4_from_Rt(R, pw) @ mat4_scale(s, s, s)
                    main.append(DrawItem(self.r.mesh_cube, M, (0.45, 0.55, 0.95, 1.0)))
                    shadow.append(DrawItem(self.r.mesh_cube, M, (0, 0, 0, 0)))

        return main, shadow, light_vp


# ==========================================================
# Teleop: VR hand → Genesis cube pose
# ==========================================================
class TeleopController:
    def __init__(self, cfg: AppConfig, handles: GenesisHandles):
        self.cfg = cfg
        self.h = handles

    def _set_entity_pos_safe(self, ent, p: np.ndarray) -> None:
        try:
            if hasattr(ent, "set_pos"):
                ent.set_pos(tuple(float(x) for x in p[:3]))
                return
        except Exception:
            pass

    def update(self, inp: InputState) -> None:
        cube = self.h.entities.get("cube")
        if cube is None:
            return

        if inp.grip and inp.hand_pose_valid and inp.hand_pos_world is not None:
            hp = np.asarray(inp.hand_pos_world, dtype=np.float32)
            gp = hp - self.cfg.scene_offset

            x = float(np.clip(gp[0], -0.6, 0.9))
            y = float(np.clip(gp[1], -0.8, 0.8))
            z = float(np.clip(0.02 + 0.30 * inp.trigger, 0.02, 0.5))

            self._set_entity_pos_safe(cube, np.array([x, y, z], dtype=np.float32))


# ==========================================================
# App loop
# ==========================================================
class App:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # 1) Genesis FIRST (avoids context mismatch issues)
        self.g = build_genesis_scene(cfg)

        # 2) XR/GL AFTER Genesis is done
        self.xr = XrRuntime(cfg)

        # Ensure context current
        sdl2.SDL_GL_MakeCurrent(self.xr.window, self.xr.gl_context)

        # 3) Create GL resources in the OpenXR-bound context
        self.renderer = Renderer(cfg)

        # 4) Spectator render target
        self.spectator = Spectator2D(size=self.cfg.spectator_tex_size)

        self.teleop = TeleopController(cfg, self.g)
        self.extractor = GenesisProxyExtractor(cfg, self.renderer, self.g)

        self._printed_fbo_warn = False

        print("[Config]")
        print("  Genesis version =", getattr(gs, "__version__", "unknown"))
        print("  Session lifecycle handling enabled (headset remove/put-back safe).")
        print("  2D spectator screen enabled =", int(self.cfg.show_screen))

    def _compute_screen_transform(self, head_R: np.ndarray, head_pos: np.ndarray) -> np.ndarray:
        """
        Build a world transform for the floating 2D screen:
          - centered in front of HMD
          - oriented facing the HMD
          - scaled to desired physical size (meters)
        """
        forward = -(head_R @ np.array([0, 0, 1], dtype=np.float32))  # HMD forward (OpenXR -Z)
        right = (head_R @ np.array([1, 0, 0], dtype=np.float32))
        up = (head_R @ np.array([0, 1, 0], dtype=np.float32))

        center = head_pos + forward * float(self.cfg.screen_distance_m) - up * float(self.cfg.screen_down_m)

        # Make a basis that faces the user:
        #   X axis = right
        #   Y axis = up
        #   Z axis = -forward (so quad's normal points toward the user)
        Rw = np.stack([right, up, -forward], axis=1)

        # Our plane mesh lies on XZ with normal +Y; rotate it so normal points +Z in its local space:
        # We can do that by swapping axes: local +Y -> world +Z; easiest: apply a fixed rotation about X:
        # rotate -90deg about X: (y->z)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        Rfinal = Rw @ Rx

        M = mat4_from_Rt(Rfinal, center) @ mat4_scale(float(self.cfg.screen_width_m), 1.0, float(self.cfg.screen_height_m))
        return M

    def run(self) -> None:
        running = True
        while running:
            running = self.xr.pump_sdl_events()
            if not running:
                break

            # Always ensure OpenXR-bound GL context is current
            if WGL.wglGetCurrentContext() != self.xr.hglrc:
                sdl2.SDL_GL_MakeCurrent(self.xr.window, self.xr.gl_context)

            # Poll OpenXR events (handles begin/end session)
            self.xr.poll_xr_events()

            if self.xr.exit_requested:
                break

            # If session is not running, do not call xrWaitFrame.
            # This is CRITICAL for correct behavior when you remove/put on headset.
            if not self.xr.session_running:
                time.sleep(0.01)
                continue

            frame_state = self.xr.wait_begin_frame()

            # Respect should_render when available (some runtimes set it false)
            if hasattr(frame_state, "should_render") and not frame_state.should_render:
                self.xr.end_frame_empty(frame_state.predicted_display_time)
                continue

            views = self.xr.locate_views(frame_state.predicted_display_time)
            if views is None:
                self.xr.end_frame_empty(frame_state.predicted_display_time)
                continue

            # Only treat input as valid when focused (recommended)
            focused = (self.xr.session_state == xr.SessionState.FOCUSED)

            if focused:
                self.xr.sync_actions()
                inp = self.xr.poll_inputs(frame_state.predicted_display_time)
            else:
                inp = InputState()

            # Teleop + physics (only if focused, so you don't get stale actions)
            if focused:
                self.teleop.update(inp)

            for _ in range(int(self.cfg.genesis_substeps_per_vr_frame)):
                self.g.scene.step()

            # Build draw lists
            main_items, shadow_items, light_vp = self.extractor.build_draw_items()

            # Shadow pass
            self.renderer.draw_shadow_pass(light_vp, shadow_items)

            # Spectator 2D pass (render to texture)
            if self.cfg.show_screen:
                self.spectator.bind()
                GL.glClearColor(0.05, 0.05, 0.06, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                spec_vp = make_spectator_vp(self.cfg)
                self.renderer.draw_main_pass(spec_vp, light_vp, main_items)
                self.spectator.unbind()

            # Render per-eye
            proj_views = []
            for eye_index, sc in enumerate(self.xr.swapchains):
                w, h = self.xr.image_sizes[eye_index]

                img_idx = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
                xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
                gl_image = self.xr.swapchain_images[eye_index][img_idx].image

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.xr.fbo)
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, gl_image, 0)

                # IMPORTANT FIX: restore draw/read buffers after shadow pass set them to NONE
                GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
                GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.xr.depth_rb)
                GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.xr.depth_rb)

                if not self._printed_fbo_warn:
                    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                    if status != GL.GL_FRAMEBUFFER_COMPLETE:
                        print("[WARN] Swapchain FBO incomplete:", hex(int(status)))
                    self._printed_fbo_warn = True

                GL.glViewport(0, 0, w, h)
                GL.glClearColor(0.07, 0.08, 0.10, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

                # Eye VP
                P = openxr_projection_from_fov(views[eye_index].fov, near=self.cfg.near, far=self.cfg.far)

                p = views[eye_index].pose.position
                q = views[eye_index].pose.orientation
                head_R = quat_xyzw_to_rotmat(np.array([q.x, q.y, q.z, q.w], dtype=np.float32))
                head_pos = np.array([p.x, p.y, p.z], dtype=np.float32)

                V = mat4_inverse_rt(head_R, head_pos)
                VP = P @ V

                # Main scene
                self.renderer.draw_main_pass(VP, light_vp, main_items)

                # Floating 2D screen (spectator texture) - stays in your app => stays focused
                if self.cfg.show_screen:
                    M_screen = self._compute_screen_transform(head_R, head_pos)
                    self.renderer.draw_screen_quad(VP, M_screen, self.spectator.tex)

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
                xr.release_swapchain_image(sc, xr.SwapchainImageReleaseInfo())

                sub_image = xr.SwapchainSubImage(
                    swapchain=sc,
                    image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(w, h)),
                    image_array_index=0,
                )
                proj_views.append(
                    xr.CompositionLayerProjectionView(
                        pose=views[eye_index].pose,
                        fov=views[eye_index].fov,
                        sub_image=sub_image,
                    )
                )

            if self.cfg.swap_eyes and len(proj_views) >= 2:
                proj_views[0], proj_views[1] = proj_views[1], proj_views[0]

            proj_layer = xr.CompositionLayerProjection(
                layer_flags=xr.CompositionLayerFlags.NONE,
                space=self.xr.reference_space,
                views=proj_views,
            )

            try:
                xr.end_frame(
                    self.xr.session,
                    xr.FrameEndInfo(
                        display_time=frame_state.predicted_display_time,
                        environment_blend_mode=self.xr.environment_blend_mode,
                        layers=[ctypes.pointer(proj_layer)],
                    ),
                )
            except Exception as e:
                # If end_frame fails due to lifecycle changes, events will drive stop/restart.
                if self.cfg.debug_inputs:
                    print("[XR][WARN] end_frame failed:", repr(e))

        self.xr.shutdown()


def main():
    if platform.system() != "Windows":
        raise RuntimeError("Windows-only (WGL + SDL OpenXR path).")

    cfg = AppConfig(
        genesis_backend="gpu",  # change to "cpu" if needed
        genesis_substeps_per_vr_frame=1,
        spectator_tex_size=1024,
        show_screen=True,
    )
    App(cfg).run()


if __name__ == "__main__":
    main()
