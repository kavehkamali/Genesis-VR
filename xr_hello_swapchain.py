"""
xr_hello_swapchain.py

Step 5 in our OpenXR + Python journey.

What it does:
- Same setup as xr_hello_frame_loop:
    - SDL2 window + OpenGL 4.x core context
    - OpenXR instance, system, session
    - LOCAL reference space
- Creates one color swapchain per eye.
- Each frame:
    - xrWaitFrame / xrBeginFrame
    - xrLocateViews (head pose per eye)
    - For each eye:
        - Acquire swapchain image
        - Clear it to a solid color (left/right different)
        - Release image
    - Submit an XrCompositionLayerProjection with both eyes.
    - xrEndFrame
- Runs ~300 frames, then exits.

You should SEE solid colors in the Quest 2 now.
"""

import ctypes
import os
import platform
import sys
import time

import xr  # pyopenxr

# SDL2 / OpenGL
if platform.system() == "Windows":
    SDL2_DIR = r"C:\SDL2"
    if os.path.isdir(SDL2_DIR):
        os.add_dll_directory(SDL2_DIR)

import sdl2
from OpenGL import WGL
from OpenGL import GL


def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def choose_swapchain_format(session):
    """Pick a GL color format supported by both runtime and GL."""
    available_formats = xr.enumerate_swapchain_formats(session)
    # Common GL formats (as ints) – not all may be present
    candidates = [
        GL.GL_SRGB8_ALPHA8,
        GL.GL_RGBA8,
    ]
    for f in available_formats:
        if f in candidates:
            print(f"Using swapchain format: 0x{f:08X}")
            return f

    # Fallback to first reported format
    print("No preferred format found, using first available:", available_formats[0])
    return available_formats[0]


def main():
    if platform.system() != "Windows":
        print("This example is written for Windows (OpenGL + WGL).")
        sys.exit(1)

    print("=== OpenXR hello_swapchain (OpenGL + SDL2 + pyopenxr) ===")

    # --------------------------------------------------------
    # 1) Enumerate instance extensions
    # --------------------------------------------------------
    exts = xr.enumerate_instance_extension_properties()
    available = set()
    for e in exts:
        name = e.extension_name
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        available.add(name)

    print("Available extensions:")
    for n in sorted(available):
        print("   ", n)

    required_ext = xr.KHR_OPENGL_ENABLE_EXTENSION_NAME
    if isinstance(required_ext, bytes):
        required_ext = required_ext.decode("utf-8", errors="ignore")

    print("\nRequired extension for OpenGL rendering:")
    print("   ", required_ext)

    if required_ext not in available:
        print(f"❌ Required {required_ext} not available. Cannot continue.")
        sys.exit(1)

    print("\n✅ XR_KHR_opengl_enable is available.")
    requested_extensions = [required_ext]

    # --------------------------------------------------------
    # 2) Create OpenXR instance
    # --------------------------------------------------------
    app_info = xr.ApplicationInfo(
        "pyopenxr_hello_swapchain",  # application_name
        1,                           # application_version
        "pyopenxr",                  # engine_name
        0,                           # engine_version
        xr.XR_CURRENT_API_VERSION,   # api_version
    )

    ici = xr.InstanceCreateInfo(
        application_info=app_info,
        enabled_extension_names=requested_extensions,
    )

    instance = xr.create_instance(ici)
    print("\n✅ Created OpenXR instance.")

    instance_props = xr.get_instance_properties(instance)
    runtime_name = to_str(instance_props.runtime_name)
    ver = instance_props.runtime_version
    print(f"Runtime name   : {runtime_name}")
    print(f"Runtime version: {ver.major}.{ver.minor}.{ver.patch}")

    # --------------------------------------------------------
    # 3) Get system + view config
    # --------------------------------------------------------
    get_info = xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY)
    system_id = xr.get_system(instance, get_info)
    print(f"\n✅ Got HMD system_id: {system_id!r} (type: {type(system_id).__name__})")

    view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
    view_config_views = xr.enumerate_view_configuration_views(
        instance, system_id, view_config_type
    )
    print(f"Number of views (eyes): {len(view_config_views)}")
    for i, v in enumerate(view_config_views):
        print(f"\nEye {i}:")
        print(f"  Recommended width : {v.recommended_image_rect_width}")
        print(f"  Recommended height: {v.recommended_image_rect_height}")
        print(f"  Max width         : {v.max_image_rect_width}")
        print(f"  Max height        : {v.max_image_rect_height}")
        print(f"  Recommended samples: {v.recommended_swapchain_sample_count}")
        print(f"  Max samples        : {v.max_swapchain_sample_count}")

    # --------------------------------------------------------
    # 4) Query OpenGL graphics requirements
    # --------------------------------------------------------
    pfn_get_gl_reqs = ctypes.cast(
        xr.get_instance_proc_addr(
            instance,
            "xrGetOpenGLGraphicsRequirementsKHR",
        ),
        xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
    )

    graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
    res = pfn_get_gl_reqs(instance, system_id, ctypes.byref(graphics_requirements))
    res = xr.exception.check_result(xr.Result(res))
    if res.is_exception():
        raise res

    print("\nOpenGL graphics requirements:")
    print("  Min API version:", graphics_requirements.min_api_version_supported)
    print("  Max API version:", graphics_requirements.max_api_version_supported)

    # --------------------------------------------------------
    # 5) Create SDL2 window + OpenGL 4.x core context
    # --------------------------------------------------------
    if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
        xr.destroy_instance(instance)
        raise RuntimeError("SDL initialization failed")

    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 0)
    sdl2.SDL_GL_SetAttribute(
        sdl2.SDL_GL_CONTEXT_PROFILE_MASK,
        sdl2.SDL_GL_CONTEXT_PROFILE_CORE,
    )

    window_width = view_config_views[0].recommended_image_rect_width // 2
    window_height = view_config_views[0].recommended_image_rect_height // 2

    print(f"\nCreating SDL2 window: {window_width} x {window_height}")
    window = sdl2.SDL_CreateWindow(
        b"OpenXR Swapchain",
        sdl2.SDL_WINDOWPOS_CENTERED,
        sdl2.SDL_WINDOWPOS_CENTERED,
        window_width,
        window_height,
        sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_SHOWN,
    )
    if not window:
        sdl2.SDL_Quit()
        xr.destroy_instance(instance)
        raise RuntimeError("Failed to create SDL window")

    gl_context = sdl2.SDL_GL_CreateContext(window)
    if not gl_context:
        sdl2.SDL_DestroyWindow(window)
        sdl2.SDL_Quit()
        xr.destroy_instance(instance)
        raise RuntimeError("Failed to create OpenGL context")

    sdl2.SDL_GL_MakeCurrent(window, gl_context)
    sdl2.SDL_GL_SetSwapInterval(0)

    gl_version = to_str(GL.glGetString(GL.GL_VERSION))
    print("✅ Created SDL2 window + OpenGL context.")
    print("OpenGL version reported by driver:", gl_version)

    # --------------------------------------------------------
    # 6) Bind OpenGL context to OpenXR
    # --------------------------------------------------------
    graphics_binding = xr.GraphicsBindingOpenGLWin32KHR()
    graphics_binding.h_dc = WGL.wglGetCurrentDC()
    graphics_binding.h_glrc = WGL.wglGetCurrentContext()

    print("WGL current DC   :", graphics_binding.h_dc)
    print("WGL current HGLRC:", graphics_binding.h_glrc)

    if not graphics_binding.h_dc or not graphics_binding.h_glrc:
        print("❌ Could not get current HDC or HGLRC from WGL.")
        sdl2.SDL_GL_DeleteContext(gl_context)
        sdl2.SDL_DestroyWindow(window)
        sdl2.SDL_Quit()
        xr.destroy_instance(instance)
        sys.exit(1)

    print("\n✅ Filled GraphicsBindingOpenGLWin32KHR with current GL context.")

    gb_ptr = ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p)
    sci = xr.SessionCreateInfo(
        0,          # create_flags
        system_id,  # system_id
        next=gb_ptr
    )

    # --------------------------------------------------------
    # 7) Create session
    # --------------------------------------------------------
    session = xr.create_session(instance, sci)
    print("✅ Created OpenXR session:", session)

    # --------------------------------------------------------
    # 8) Create reference space (LOCAL)
    # --------------------------------------------------------
    identity_orientation = xr.Quaternionf(0.0, 0.0, 0.0, 1.0)
    identity_position = xr.Vector3f(0.0, 0.0, 0.0)
    identity_pose = xr.Posef(
        orientation=identity_orientation,
        position=identity_position,
    )

    ref_space_info = xr.ReferenceSpaceCreateInfo(
        reference_space_type=xr.ReferenceSpaceType.LOCAL,
        pose_in_reference_space=identity_pose,
    )
    reference_space = xr.create_reference_space(session, ref_space_info)
    print("✅ Created LOCAL reference space:", reference_space)

    # --------------------------------------------------------
    # 9) Create color swapchains (one per eye)
    # --------------------------------------------------------
    color_format = choose_swapchain_format(session)

    swapchains = []
    swapchain_images = []  # list of lists of SwapchainImageOpenGLKHR
    image_sizes = []

    for i, v in enumerate(view_config_views):
        width = v.recommended_image_rect_width
        height = v.recommended_image_rect_height
        sample_count = v.recommended_swapchain_sample_count or 1

        sci = xr.SwapchainCreateInfo(
            create_flags=0,
            usage_flags=(
                xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                | xr.SwapchainUsageFlags.SAMPLED_BIT
            ),
            format=color_format,
            sample_count=sample_count,
            width=width,
            height=height,
            face_count=1,
            array_size=1,
            mip_count=1,
        )
        sc = xr.create_swapchain(session, sci)
        print(f"✅ Created swapchain for eye {i}: {sc}")

        # Enumerate GL images for this swapchain
        images = xr.enumerate_swapchain_images(
            sc, xr.SwapchainImageOpenGLKHR
        )
        print(f"   Swapchain {i} has {len(images)} images.")
        swapchains.append(sc)
        swapchain_images.append(images)
        image_sizes.append((width, height))

    # Create a simple framebuffer for rendering into textures
    fbo = GL.glGenFramebuffers(1)

    # --------------------------------------------------------
    # 10) Begin XR session
    # --------------------------------------------------------
    begin_info = xr.SessionBeginInfo(
        primary_view_configuration_type=view_config_type
    )
    xr.begin_session(session, begin_info)
    print("✅ Began XR session.")

    # --------------------------------------------------------
    # 11) Frame loop with rendering and projection layer
    # --------------------------------------------------------
    print("\n=== Entering frame loop (about 300 frames) ===")
    environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE

    frame_limit = 300
    frame_index = 0

    sdl_event = sdl2.SDL_Event()

    while frame_index < frame_limit:
        # Handle SDL window events
        while sdl2.SDL_PollEvent(ctypes.byref(sdl_event)) != 0:
            if sdl_event.type == sdl2.SDL_QUIT:
                frame_index = frame_limit
                break

        frame_state = xr.wait_frame(session)
        xr.begin_frame(session)

        # Locate views (head pose)
        locate_info = xr.ViewLocateInfo(
            view_configuration_type=view_config_type,
            display_time=frame_state.predicted_display_time,
            space=reference_space,
        )
        view_state, located_views = xr.locate_views(session, locate_info)

        # Build projection views / render to swapchains
        proj_views = []
        layer_flags = xr.SwapchainUsageFlags.NONE  # not really used here

        for eye_index, sc in enumerate(swapchains):
            w, h = image_sizes[eye_index]

            # Acquire image index
            acquire_info = xr.SwapchainImageAcquireInfo()
            image_index = xr.acquire_swapchain_image(sc, acquire_info)

            # Wait for it – use INFINITE_DURATION so we don't timeout
            wait_info = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
            xr.wait_swapchain_image(sc, wait_info)

            # Get the GL texture for this image index
            gl_image = swapchain_images[eye_index][image_index].image

            # Render into texture: bind FBO, attach texture, clear
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D,
                gl_image,
                0,
            )
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                print(f"⚠ Incomplete framebuffer for eye {eye_index}, status=0x{status:X}")

            GL.glViewport(0, 0, w, h)

            # Different clear color per eye so you can tell them apart
            if eye_index == 0:
                GL.glClearColor(0.0, 0.0, 1.0, 1.0)  # blue
            else:
                GL.glClearColor(1.0, 0.0, 0.0, 1.0)  # red

            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # Unbind framebuffer
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

            # Release image back to runtime
            release_info = xr.SwapchainImageReleaseInfo()
            xr.release_swapchain_image(sc, release_info)

            # Build projection view for this eye
            sub_image = xr.SwapchainSubImage(
                swapchain=sc,
                image_rect=xr.Rect2Di(
                    offset=xr.Offset2Di(0, 0),
                    extent=xr.Extent2Di(w, h),
                ),
                image_array_index=0,
            )

            proj_view = xr.CompositionLayerProjectionView(
                pose=located_views[eye_index].pose,
                fov=located_views[eye_index].fov,
                sub_image=sub_image,
            )
            proj_views.append(proj_view)

        # Projection layer for both eyes
        proj_layer = xr.CompositionLayerProjection(
            layer_flags=layer_flags,
            space=reference_space,
            views=proj_views,
        )

        # Submit frame with our single projection layer.
        # IMPORTANT: pyopenxr expects pointers to layer structs, not structs directly.
        layers = [ctypes.pointer(proj_layer)]

        frame_end_info = xr.FrameEndInfo(
            display_time=frame_state.predicted_display_time,
            environment_blend_mode=environment_blend_mode,
            layers=layers,
        )
        xr.end_frame(session, frame_end_info)


        frame_index += 1
        time.sleep(0.005)

    print("\nExiting frame loop.")

    # --------------------------------------------------------
    # 12) End session & cleanup
    # --------------------------------------------------------
    try:
        xr.end_session(session)
        print("✅ Ended XR session.")
    except Exception as e:
        # Avoid noise from SessionNotStoppingError etc.
        print("⚠ xr.end_session raised (ignored for cleanup):", repr(e))

    print("Cleaning up...")
    xr.destroy_session(session)
    xr.destroy_instance(instance)
    GL.glDeleteFramebuffers(1, [fbo])
    sdl2.SDL_GL_DeleteContext(gl_context)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()
    print("✅ Clean exit.")


if __name__ == "__main__":
    main()
