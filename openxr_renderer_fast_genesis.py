import ctypes
import numpy as np
import OpenGL.GL as gl
import pyglet
import xr  # pyopenxr

import genesis as gs
from genesis.repr_base import RBC
from genesis.ext import pyrender
from genesis.ext.pyrender.constants import RenderFlags
from genesis.ext.pyrender.renderer import Renderer


class OpenXRRendererFastGenesis(RBC):
    """
    Genesis -> OpenXR (Meta Quest Link) GPU->GPU renderer.
    Renders into OpenXR swapchain textures via an FBO (no CPU readback).
    """

    def __init__(self, context, resolution_scale=1.0):
        self._context = context
        self._resolution_scale = float(resolution_scale)

        self._oxr = None          # genesis.ext.pyrender.OffscreenRenderer (GL context owner)
        self._renderer = None     # genesis.ext.pyrender.Renderer

        self._instance = None
        self._system_id = None
        self._session = None
        self._space = None
        self._view_type = None
        self._blend_mode = None

        self._running = False
        self._should_quit = False

        self._swap = []  # per eye: {"sc","w","h","images"}
        self._fbos = [gl.glGenFramebuffers(1), gl.glGenFramebuffers(1)]
        self._depth = [gl.glGenRenderbuffers(1), gl.glGenRenderbuffers(1)]

        self._cam_nodes = {}

    @property
    def should_quit(self):
        return self._should_quit

    def build(self):
        # Create an OpenGL context using Genesis' offscreen (pyglet) platform
        self._oxr = pyrender.OffscreenRenderer(pyopengl_platform="pyglet", seg_node_map=self._context.seg_node_map)
        self._oxr.make_current()

        # Init OpenXR on the *current* WGL context
        self._init_openxr()

        # Create swapchains + depth buffers
        self._create_swapchains()

        # Genesis renderer sized to the left eye
        w0, h0 = self._swap[0]["w"], self._swap[0]["h"]
        self._renderer = Renderer(w0, h0, self._context.jit, point_size=1.0)

        # Two static cameras (one per eye). No motion/head tracking for now.
        for eye in (0, 1):
            cam = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(70.0),
                znear=0.05,
                zfar=100.0,
                aspectRatio=1.0,
            )
            node = self._context.add_node(cam)
            self._cam_nodes[eye] = node
            self._context.set_node_pose(node, np.eye(4, dtype=np.float32))

        self._oxr.make_uncurrent()

    def poll_events(self):
        while True:
            try:
                ev = xr.poll_event(self._instance)
            except xr.exception.EventUnavailable:
                break

            if isinstance(ev, xr.EventDataSessionStateChanged):
                state = ev.state

                # Optional: helpful logging
                # print("[XR] Session state ->", getattr(state, "name", state))

                if state == xr.SessionState.READY and not self._running:
                    xr.begin_session(
                        self._session,
                        xr.SessionBeginInfo(primary_view_configuration_type=self._view_type),
                    )
                    self._running = True

                elif state == xr.SessionState.STOPPING and self._running:
                    xr.end_session(self._session)
                    self._running = False
                    self._should_quit = True

                elif state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                    self._should_quit = True

    def update_scene(self, force_render=True):
        self._context.update(force_render)

    def render_frame(self):
        if not self._running or self._should_quit:
            return

        frame_state = xr.wait_frame(self._session, xr.FrameWaitInfo())
        xr.begin_frame(self._session, xr.FrameBeginInfo())

        view_state, views = xr.locate_views(
            self._session,
            xr.ViewLocateInfo(
                view_configuration_type=self._view_type,
                display_time=frame_state.predicted_display_time,
                space=self._space,
            ),
            2,
        )

        self._oxr.make_current()

        # Update Genesis GPU buffers once per frame
        self._context.jit.update_buffer(self._context.buffer)
        self._context.buffer.clear()

        proj_views = [xr.CompositionLayerProjectionView(), xr.CompositionLayerProjectionView()]

        for eye in (0, 1):
            sc = self._swap[eye]["sc"]
            w = self._swap[eye]["w"]
            h = self._swap[eye]["h"]
            images = self._swap[eye]["images"]

            idx = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
            color_tex = images[idx].image  # GLuint texture id

            # Bind swapchain image into an FBO
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbos[eye])
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_tex, 0)
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self._depth[eye]
            )

            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Swapchain FBO incomplete eye {eye}: {hex(status)}")

            gl.glViewport(0, 0, w, h)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # Static camera pose (identity). No motion.
            node = self._cam_nodes[eye]
            self._context.set_node_pose(node, np.eye(4, dtype=np.float32))

            # GPU->GPU render into CURRENT FBO (requires your OffscreenRenderer.render_to_current_fbo patch)
            self._oxr.render_to_current_fbo(
                self._context._scene,
                self._renderer,
                camera_node=node,
                flags=RenderFlags.NONE,
                shadow=False,
                plane_reflection=False,
                env_separate_rigid=self._context.env_separate_rigid,
            )

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

            xr.release_swapchain_image(sc, xr.SwapchainImageReleaseInfo())

            pv = proj_views[eye]
            pv.pose = views[eye].pose
            pv.fov = views[eye].fov
            pv.sub_image.swapchain = sc
            pv.sub_image.image_rect.offset = xr.Offset2Di(0, 0)
            pv.sub_image.image_rect.extent = xr.Extent2Di(w, h)

        self._oxr.make_uncurrent()

        layer = xr.CompositionLayerProjection()
        layer.space = self._space
        layer.views = proj_views

        xr.end_frame(
            self._session,
            xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=self._blend_mode,
                layers=[layer],
            ),
        )

    def destroy(self):
        try:
            if self._oxr:
                self._oxr.make_current()
        except Exception:
            pass

        try:
            gl.glDeleteFramebuffers(1, [self._fbos[0]])
            gl.glDeleteFramebuffers(1, [self._fbos[1]])
            gl.glDeleteRenderbuffers(1, [self._depth[0]])
            gl.glDeleteRenderbuffers(1, [self._depth[1]])
        except Exception:
            pass

        try:
            if self._session:
                xr.destroy_session(self._session)
        except Exception:
            pass
        try:
            if self._instance:
                xr.destroy_instance(self._instance)
        except Exception:
            pass

        try:
            if self._renderer:
                self._renderer.delete()
        except Exception:
            pass

        try:
            if self._oxr:
                self._oxr.delete()
        except Exception:
            pass

    def _init_openxr(self):
        # Ensure extension is present
        exts = xr.enumerate_instance_extension_properties()
        if xr.KHR_OPENGL_ENABLE_EXTENSION_NAME not in exts:
            raise RuntimeError("OpenXR runtime does not advertise XR_KHR_opengl_enable.")

        # Instance
        self._instance = xr.create_instance(
            xr.InstanceCreateInfo(
                application_info=xr.ApplicationInfo(
                    "GenesisPyOpenXR",
                    1,
                    "Genesis",
                    1,
                    xr.XR_CURRENT_API_VERSION,
                ),
                enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            )
        )

        # System
        self._system_id = xr.get_system(
            self._instance,
            xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY),
        )

        self._view_type = xr.ViewConfigurationType.PRIMARY_STEREO
        modes = xr.enumerate_environment_blend_modes(self._instance, self._system_id, self._view_type)
        self._blend_mode = modes[0] if modes else xr.EnvironmentBlendMode.OPAQUE

        # ---- OpenGL graphics requirements (matches your working script) ----
        pfn_get_gl_reqs = ctypes.cast(
            xr.get_instance_proc_addr(self._instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )

        gl_reqs = xr.GraphicsRequirementsOpenGLKHR()
        res = pfn_get_gl_reqs(self._instance, self._system_id, ctypes.byref(gl_reqs))
        res = xr.exception.check_result(xr.Result(res))
        if res.is_exception():
            raise res

        # Print helpful info (optional but useful)
        try:
            gl_ver = gl.glGetString(gl.GL_VERSION)
            gl_ver = gl_ver.decode("utf-8", errors="ignore") if isinstance(gl_ver, (bytes, bytearray)) else str(gl_ver)
        except Exception:
            gl_ver = "<unknown>"
        print("[Genesis/OpenXR] GL_VERSION:", gl_ver)
        print("[Genesis/OpenXR] GL reqs min:", gl_reqs.min_api_version_supported, "max:", gl_reqs.max_api_version_supported)

        # ---- WGL handles from CURRENT context ----
        hdc, hglrc = self._get_wgl_handles_from_current_pyglet()

        # ---- Graphics binding: IMPORTANT field names ----
        binding = xr.GraphicsBindingOpenGLWin32KHR()

        # pyopenxr uses snake_case fields (as in your SDL runtime)
        if hasattr(binding, "h_dc"):
            binding.h_dc = hdc
        if hasattr(binding, "h_glrc"):
            binding.h_glrc = hglrc

        # Some builds might expose camelCase too; set both defensively
        if hasattr(binding, "hDC"):
            binding.hDC = ctypes.c_void_p(hdc)
        if hasattr(binding, "hGLRC"):
            binding.hGLRC = ctypes.c_void_p(hglrc)

        # ---- SessionCreateInfo: match your working script style ----
        gb_ptr = ctypes.cast(ctypes.pointer(binding), ctypes.c_void_p)
        sci = xr.SessionCreateInfo(
            0,
            self._system_id,
            next=gb_ptr,
        )

        self._session = xr.create_session(self._instance, sci)

        # Reference space (LOCAL)
        identity_pose = xr.Posef(
            orientation=xr.Quaternionf(0.0, 0.0, 0.0, 1.0),
            position=xr.Vector3f(0.0, 0.0, 0.0),
        )
        self._space = xr.create_reference_space(
            self._session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL,
                pose_in_reference_space=identity_pose,
            ),
        )

    def _create_swapchains(self):
        cfgs = xr.enumerate_view_configuration_views(self._instance, self._system_id, self._view_type)
        if len(cfgs) < 2:
            raise RuntimeError("Stereo view configuration not available.")

        formats = xr.enumerate_swapchain_formats(self._session)
        GL_SRGB8_ALPHA8 = 0x8C43
        GL_RGBA8 = 0x8058
        fmt = GL_SRGB8_ALPHA8 if GL_SRGB8_ALPHA8 in formats else (GL_RGBA8 if GL_RGBA8 in formats else formats[0])

        self._swap.clear()
        for eye in (0, 1):
            w = int(cfgs[eye].recommended_image_rect_width * self._resolution_scale)
            h = int(cfgs[eye].recommended_image_rect_height * self._resolution_scale)

            sc = xr.create_swapchain(
                self._session,
                xr.SwapchainCreateInfo(
                    usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT,
                    format=fmt,
                    sample_count=cfgs[eye].recommended_swapchain_sample_count,
                    width=w,
                    height=h,
                    face_count=1,
                    array_size=1,
                    mip_count=1,
                ),
            )
            images = xr.enumerate_swapchain_images(sc, xr.SwapchainImageOpenGLKHR)

            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth[eye])
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, w, h)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

            self._swap.append({"sc": sc, "w": w, "h": h, "images": images})

    def _get_wgl_handles_from_current_pyglet(self):
        import ctypes
        from ctypes import wintypes

        opengl32 = ctypes.WinDLL("opengl32", use_last_error=True)
        opengl32.wglGetCurrentContext.restype = wintypes.HANDLE
        opengl32.wglGetCurrentDC.restype = wintypes.HANDLE

        hglrc = opengl32.wglGetCurrentContext()
        hdc = opengl32.wglGetCurrentDC()

        if not hglrc or not hdc:
            raise RuntimeError("No current WGL context/DC. OffscreenRenderer.make_current() must be active on this thread.")
        return int(hdc), int(hglrc)


