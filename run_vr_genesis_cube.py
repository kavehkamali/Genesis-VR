# run_vr_opengl_game_style_shadows.py
#
# Fixes:
# 1) Adds color + texture + lighting + shadows (shadow map)
# 2) Correct camera pose: view = inverse(OpenXR eye pose)
# 3) Fixed objects stay fixed in reference space
# 4) Correct stereo: DO NOT scale eye separation (no IPD scaling)
#
# Still uses the fast pipeline: render directly into OpenXR swapchain textures (GPU->GPU).
#
# Controls:
#   - Hold Grip -> move ACTIVE cube with right-hand pose
#   - Button A -> RED
#   - Button B -> BLUE
#   - Trigger > 0.2 -> GREEN
#   - Thumbstick beyond deadzone -> YELLOW
#   - Else -> GRAY

import ctypes
import os
import platform
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
# USER TUNABLES
# ==========================================================
# Place the scene a bit down and forward in XR space (XR: +X right, +Y up, -Z forward)
SCENE_OFFSET = np.array([0.0, -1.2, -1.2], dtype=np.float32)

STICK_DEADZONE = 0.30
DEBUG_INPUTS = True

SWAP_EYES = False  # keep False unless you know your runtime is swapped

# Shadows
SHADOW_MAP_SIZE = 2048
LIGHT_DIR = np.array([0.35, -1.0, 0.25], dtype=np.float32)  # directional light (world space)
LIGHT_DIR = LIGHT_DIR / (np.linalg.norm(LIGHT_DIR) + 1e-9)

# Scene layout (similar to your Genesis scene, in "scene local" coords)
RING_Z = 0.10
RING_OUTER = 1.0
HOLE = 0.35
PLATE_T = 0.05
RAIL_W = (RING_OUTER - HOLE) / 2.0

CUBE_SIZE = 0.08
START_POS_LOCAL = np.array([0.30, 0.0, RING_Z + 0.25], dtype=np.float32)

COLORS = {
    "gray":   (0.6, 0.6, 0.6, 1.0),
    "red":    (1.0, 0.1, 0.1, 1.0),
    "blue":   (0.1, 0.3, 1.0, 1.0),
    "green":  (0.2, 1.0, 0.2, 1.0),
    "yellow": (1.0, 1.0, 0.2, 1.0),
}
COLOR_NAMES = ["gray", "red", "blue", "green", "yellow"]


# ==========================================================
# Math helpers
# ==========================================================
def normalize(v):
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n

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

def mat4_identity():
    return np.eye(4, dtype=np.float32)

def mat4_translate(t):
    M = mat4_identity()
    M[0, 3] = float(t[0])
    M[1, 3] = float(t[1])
    M[2, 3] = float(t[2])
    return M

def mat4_scale(sx, sy, sz):
    M = mat4_identity()
    M[0, 0] = float(sx)
    M[1, 1] = float(sy)
    M[2, 2] = float(sz)
    return M

def mat4_inverse_rt(R, t):
    Rt = R.T
    ti = -Rt @ t
    M = mat4_identity()
    M[:3, :3] = Rt
    M[:3, 3] = ti
    return M

def openxr_projection_from_fov(fov: xr.Fovf, near=0.05, far=50.0):
    # OpenGL clip z in [-1, 1]
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

def look_at(eye, target, up):
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


# ==========================================================
# OpenGL shader helpers
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


# ==========================================================
# Renderer with texture + lighting + shadow map
# ==========================================================
class LitShadowRenderer:
    def __init__(self):
        self._init_programs()
        self._init_meshes()
        self._init_checker_texture()
        self._init_shadow_map()

    def _init_programs(self):
        # Main shader: textured + colored + lambert + shadow
        vs_src = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec3 aNrm;
        layout(location=2) in vec2 aUV;

        uniform mat4 uM;
        uniform mat4 uVP;
        uniform mat4 uLightVP;

        out vec3 vWorldPos;
        out vec3 vNrm;
        out vec2 vUV;
        out vec4 vShadowCoord;

        void main() {
            vec4 wp = uM * vec4(aPos, 1.0);
            vWorldPos = wp.xyz;
            vNrm = mat3(uM) * aNrm;
            vUV = aUV;
            vShadowCoord = uLightVP * wp;
            gl_Position = uVP * wp;
        }
        """

        fs_src = r"""
        #version 330 core
        in vec3 vWorldPos;
        in vec3 vNrm;
        in vec2 vUV;
        in vec4 vShadowCoord;

        uniform vec4 uColor;
        uniform vec3 uLightDir;     // normalized, world space
        uniform sampler2D uTex;
        uniform sampler2DShadow uShadowMap;

        out vec4 oColor;

        float shadow_factor(vec4 sc) {
            // Perspective divide to NDC
            vec3 proj = sc.xyz / sc.w;
            // Map from [-1,1] to [0,1]
            vec3 uvz = proj * 0.5 + 0.5;

            // Outside shadow map => lit
            if (uvz.x < 0.0 || uvz.x > 1.0 || uvz.y < 0.0 || uvz.y > 1.0) return 1.0;

            // Depth compare with small bias
            float bias = 0.0025;
            // sampler2DShadow expects vec3(uv, depth)
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

        # Shadow pass: depth only
        shadow_vs = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uM;
        uniform mat4 uLightVP;
        void main() {
            gl_Position = uLightVP * (uM * vec4(aPos, 1.0));
        }
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

        # Uniform locations
        self.uM = GL.glGetUniformLocation(self.prog, "uM")
        self.uVP = GL.glGetUniformLocation(self.prog, "uVP")
        self.uLightVP = GL.glGetUniformLocation(self.prog, "uLightVP")
        self.uColor = GL.glGetUniformLocation(self.prog, "uColor")
        self.uLightDir = GL.glGetUniformLocation(self.prog, "uLightDir")
        self.uTex = GL.glGetUniformLocation(self.prog, "uTex")
        self.uShadowMap = GL.glGetUniformLocation(self.prog, "uShadowMap")

        self.s_uM = GL.glGetUniformLocation(self.shadow_prog, "uM")
        self.s_uLightVP = GL.glGetUniformLocation(self.shadow_prog, "uLightVP")

    def _init_meshes(self):
        # Cube and plane with normals+uv
        self.vao_cube, self.vbo_cube, self.cube_count = self._make_cube()
        self.vao_plane, self.vbo_plane, self.plane_count = self._make_plane()

    def _make_cube(self):
        # Position, Normal, UV (36 verts)
        # UVs are simple; good enough for checker
        data = []

        def add_face(n, verts, uvs):
            for (p, uv) in zip(verts, uvs):
                data.extend([p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]])

        # Each face: two triangles (6 verts)
        # +Z
        add_face((0,0,1),
                 [(-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,0.5,0.5),
                  (-0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5)],
                 [(0,0),(1,0),(1,1),(0,0),(1,1),(0,1)])
        # -Z
        add_face((0,0,-1),
                 [(-0.5,-0.5,-0.5),(0.5,0.5,-0.5),(0.5,-0.5,-0.5),
                  (-0.5,-0.5,-0.5),(-0.5,0.5,-0.5),(0.5,0.5,-0.5)],
                 [(0,0),(1,1),(1,0),(0,0),(0,1),(1,1)])
        # +X
        add_face((1,0,0),
                 [(0.5,-0.5,-0.5),(0.5,0.5,0.5),(0.5,-0.5,0.5),
                  (0.5,-0.5,-0.5),(0.5,0.5,-0.5),(0.5,0.5,0.5)],
                 [(0,0),(1,1),(1,0),(0,0),(0,1),(1,1)])
        # -X
        add_face((-1,0,0),
                 [(-0.5,-0.5,-0.5),(-0.5,-0.5,0.5),(-0.5,0.5,0.5),
                  (-0.5,-0.5,-0.5),(-0.5,0.5,0.5),(-0.5,0.5,-0.5)],
                 [(0,0),(1,0),(1,1),(0,0),(1,1),(0,1)])
        # +Y
        add_face((0,1,0),
                 [(-0.5,0.5,-0.5),(-0.5,0.5,0.5),(0.5,0.5,0.5),
                  (-0.5,0.5,-0.5),(0.5,0.5,0.5),(0.5,0.5,-0.5)],
                 [(0,0),(0,1),(1,1),(0,0),(1,1),(1,0)])
        # -Y
        add_face((0,-1,0),
                 [(-0.5,-0.5,-0.5),(0.5,-0.5,0.5),(-0.5,-0.5,0.5),
                  (-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),(0.5,-0.5,0.5)],
                 [(0,0),(1,1),(0,1),(0,0),(1,0),(1,1)])

        v = np.array(data, dtype=np.float32)

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

        return vao, vbo, int(len(v) / 8)

    def _make_plane(self):
        # Unit plane on XZ (y=0), normals up, UV tiled
        v = np.array([
            # pos            nrm        uv
            -0.5,0.0,-0.5,   0,1,0,      0,0,
             0.5,0.0,-0.5,   0,1,0,      4,0,
             0.5,0.0, 0.5,   0,1,0,      4,4,

            -0.5,0.0,-0.5,   0,1,0,      0,0,
             0.5,0.0, 0.5,   0,1,0,      4,4,
            -0.5,0.0, 0.5,   0,1,0,      0,4,
        ], dtype=np.float32)

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

        return vao, vbo, int(len(v) / 8)

    def _init_checker_texture(self):
        # Simple checkerboard texture (RGB)
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

    def _init_shadow_map(self):
        self.shadow_fbo = GL.glGenFramebuffers(1)
        self.shadow_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24,
                        SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0,
                        GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_INT, None)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        # Important for sampler2DShadow
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

    def draw_shadow_pass(self, light_vp, draw_calls):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_fbo)
        GL.glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glCullFace(GL.GL_FRONT)  # helps reduce shadow acne
        GL.glEnable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.shadow_prog)
        GL.glUniformMatrix4fv(self.s_uLightVP, 1, True, light_vp)

        for (mesh, M) in draw_calls:
            GL.glUniformMatrix4fv(self.s_uM, 1, True, M)
            vao, count = mesh
            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, count)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def draw_main_pass(self, vp, light_vp, draw_calls):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glCullFace(GL.GL_BACK)
        GL.glEnable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.uVP, 1, True, vp)
        GL.glUniformMatrix4fv(self.uLightVP, 1, True, light_vp)
        GL.glUniform3f(self.uLightDir, float(LIGHT_DIR[0]), float(LIGHT_DIR[1]), float(LIGHT_DIR[2]))

        # Bind checker texture
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.checker_tex)
        GL.glUniform1i(self.uTex, 0)

        # Bind shadow map
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_tex)
        GL.glUniform1i(self.uShadowMap, 1)

        for (mesh, M, color) in draw_calls:
            GL.glUniformMatrix4fv(self.uM, 1, True, M)
            GL.glUniform4f(self.uColor, float(color[0]), float(color[1]), float(color[2]), float(color[3]))
            vao, count = mesh
            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, count)

        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_CULL_FACE)

    @property
    def mesh_cube(self):
        return (self.vao_cube, self.cube_count)

    @property
    def mesh_plane(self):
        return (self.vao_plane, self.plane_count)


# ==========================================================
# OpenXR runtime
# ==========================================================
class XrRuntime:
    def __init__(self, title="VR OpenGL Game-Style + Shadows"):
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

        app_info = xr.ApplicationInfo("vr_opengl_game_style_shadows", 1, "pyopenxr", 0, xr.XR_CURRENT_API_VERSION)
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

        # Prefer STAGE if available (stable world)
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

    def dbg_inputs(self, grip, trigger, a, b, sx, sy, pose_valid):
        if not DEBUG_INPUTS:
            return
        last = self._dbg_last
        def pr(name, val): print(f"[DBG] {name}: {val}")

        if last["grip"] is None or grip != last["grip"]: pr("Grip", int(grip))
        if last["a"] is None or a != last["a"]: pr("Button A", int(a))
        if last["b"] is None or b != last["b"]: pr("Button B", int(b))
        if last["trig"] is None or abs(trigger - last["trig"]) > 0.05: pr("Trigger", f"{trigger:.2f}")
        if last["sx"] is None or abs(sx - last["sx"]) > 0.20: pr("Thumbstick X", f"{sx:.2f}")
        if last["sy"] is None or abs(sy - last["sy"]) > 0.20: pr("Thumbstick Y", f"{sy:.2f}")
        if last["pose_valid"] is None or pose_valid != last["pose_valid"]: pr("Hand pose valid", int(pose_valid))

        last["grip"] = grip
        last["a"] = a
        last["b"] = b
        last["trig"] = trigger
        last["sx"] = sx
        last["sy"] = sy
        last["pose_valid"] = pose_valid


# ==========================================================
# Sim state
# ==========================================================
class SimState:
    def __init__(self):
        self.active_idx = 0
        self.last_cube_pos_local = START_POS_LOCAL.copy()
        self.trigger = 0.0

    def _choose_color(self, button_a, button_b, stick_active, trigger):
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

    def update(self, grip, trigger, button_a, button_b, sx, sy, hand_pos_world):
        stick_active = (abs(sx) > STICK_DEADZONE) or (abs(sy) > STICK_DEADZONE)
        self.active_idx = self._choose_color(button_a, button_b, stick_active, trigger)
        self.trigger = trigger

        if grip and hand_pos_world is not None:
            # Convert hand world position into our scene-local by subtracting SCENE_OFFSET
            hp = np.array(hand_pos_world, dtype=np.float32) - SCENE_OFFSET

            # Keep same "feel" as your Genesis example: constrain X/Y and drive Z from trigger
            x = float(np.clip(hp[0], -0.45, 0.45))
            y = float(np.clip(hp[1], -0.45, 0.45))
            z = float(RING_Z + 0.12 + 0.25 * trigger)

            self.last_cube_pos_local[:] = (x, y, z)


# ==========================================================
# Scene draw list builders
# ==========================================================
def build_scene_draw_calls(renderer: LitShadowRenderer, sim: SimState):
    # Build object transforms in WORLD space (XR reference space), by adding SCENE_OFFSET.
    calls_main = []
    calls_shadow = []

    def world_M_from_local(center_local, size_xyz):
        center_world = SCENE_OFFSET + np.array(center_local, dtype=np.float32)
        return mat4_translate(center_world) @ mat4_scale(size_xyz[0], size_xyz[1], size_xyz[2])

    # Big ground plane (world)
    # Put plane under scene, XZ plane, y = SCENE_OFFSET.y - 0.05
    plane_center = np.array([0.0, -0.05, 0.0], dtype=np.float32) + SCENE_OFFSET
    M_plane = mat4_translate(plane_center) @ mat4_scale(8.0, 1.0, 8.0)  # y scale unused for plane but ok
    calls_main.append((renderer.mesh_plane, M_plane, (0.85, 0.85, 0.85, 1.0)))
    calls_shadow.append((renderer.mesh_plane, M_plane))

    # Rails (scaled cubes)
    ring_z = RING_Z
    ring_outer = RING_OUTER
    hole = HOLE
    plate_t = PLATE_T
    rail_w = RAIL_W

    rail_color = (0.55, 0.55, 0.60, 1.0)

    rails = [
        ((0.0, -(hole/2.0 + rail_w/2.0), ring_z), (ring_outer, rail_w, plate_t)),
        ((0.0, +(hole/2.0 + rail_w/2.0), ring_z), (ring_outer, rail_w, plate_t)),
        ((-(hole/2.0 + rail_w/2.0), 0.0, ring_z), (rail_w, hole, plate_t)),
        ((+(hole/2.0 + rail_w/2.0), 0.0, ring_z), (rail_w, hole, plate_t)),
    ]
    for c, s in rails:
        M = world_M_from_local(c, s)
        calls_main.append((renderer.mesh_cube, M, rail_color))
        calls_shadow.append((renderer.mesh_cube, M))

    # Active cube
    cube_color = COLORS[COLOR_NAMES[sim.active_idx]]
    cube_center_world = SCENE_OFFSET + sim.last_cube_pos_local
    M_cube = mat4_translate(cube_center_world) @ mat4_scale(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    calls_main.append((renderer.mesh_cube, M_cube, cube_color))
    calls_shadow.append((renderer.mesh_cube, M_cube))

    return calls_main, calls_shadow


def compute_light_vp():
    # Directional light: build an orthographic light view around the scene area.
    # Center around SCENE_OFFSET.
    center = SCENE_OFFSET + np.array([0.0, 0.0, 0.4], dtype=np.float32)
    light_pos = center - LIGHT_DIR * 4.0
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(up, LIGHT_DIR)) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    V = look_at(light_pos, center, up)

    # Ortho bounds big enough for the ring area
    l, r = -2.5, 2.5
    b, t = -2.5, 2.5
    n, f = 0.1, 10.0

    P = np.array([
        [2/(r-l), 0,       0,       -(r+l)/(r-l)],
        [0,       2/(t-b), 0,       -(t+b)/(t-b)],
        [0,       0,      -2/(f-n), -(f+n)/(f-n)],
        [0,       0,       0,        1],
    ], dtype=np.float32)

    return P @ V


# ==========================================================
# Main
# ==========================================================
def main():
    if platform.system() != "Windows":
        raise RuntimeError("Windows-only (WGL + SDL).")

    rt = XrRuntime("VR OpenGL Game-Style + Shadows (Fixed Stereo)")
    renderer = LitShadowRenderer()
    sim = SimState()

    print("[Config]")
    print("  SCENE_OFFSET =", tuple(float(x) for x in SCENE_OFFSET))
    print("  SWAP_EYES =", SWAP_EYES)
    print("  DEBUG_INPUTS =", DEBUG_INPUTS)
    print("  NOTE: Stereo is correct ONLY if you do NOT scale IPD (we don't).")

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

        rt.dbg_inputs(grip, trigger, button_a, button_b, sx, sy, pose_valid)

        # Views (per-eye pose + fov from runtime)
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

        # Update sim
        sim.update(grip, trigger, button_a, button_b, sx, sy, hand_pos)

        # Build draw calls and shadow matrix
        light_vp = compute_light_vp()
        calls_main, calls_shadow = build_scene_draw_calls(renderer, sim)

        # Shadow pass once per frame (shared for both eyes)
        renderer.draw_shadow_pass(light_vp, calls_shadow)

        proj_views = []
        for eye_index, sc in enumerate(rt.swapchains):
            w, h = rt.image_sizes[eye_index]

            img_idx = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
            gl_image = rt.swapchain_images[eye_index][img_idx].image

            # Bind swapchain texture to FBO
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, rt.fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, gl_image, 0)

            # Depth buffer for main render
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rt.depth_rb)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rt.depth_rb)

            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.07, 0.08, 0.10, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Projection from OpenXR per-eye fov
            P = openxr_projection_from_fov(located_views[eye_index].fov, near=0.05, far=50.0)

            # View: inverse of eye pose in reference space (this is the big fix for "world moves")
            p = located_views[eye_index].pose.position
            q = located_views[eye_index].pose.orientation
            R = quat_to_rotmat_xyzw(q.x, q.y, q.z, q.w)
            tpos = np.array([p.x, p.y, p.z], dtype=np.float32)

            V = mat4_inverse_rt(R, tpos)
            VP = P @ V

            # Main draw
            renderer.draw_main_pass(VP, light_vp, calls_main)

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
