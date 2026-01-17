"""
xr_hello_frame_loop.py

Step 4 in our OpenXR + Python journey.

What it does:
- Same setup as xr_hello_session: SDL2 window + OpenGL 4.x core context + OpenXR session.
- Creates a LOCAL reference space.
- Starts an XR session.
- Runs a simple frame loop (~300 frames):
    - xrWaitFrame
    - xrBeginFrame
    - xrLocateViews (gets head pose per eye)
    - xrEndFrame (no layers yet)
- Logs head pose for eye 0 each frame.
- Cleans up and exits.

Still no actual rendering in the headset – that’s Step 5.
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


def main():
    if platform.system() != "Windows":
        print("This example is written for Windows (OpenGL + WGL).")
        sys.exit(1)

    print("=== OpenXR hello_frame_loop (OpenGL + SDL2 + pyopenxr) ===")

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
        "pyopenxr_hello_frame_loop",  # application_name
        1,                            # application_version
        "pyopenxr",                   # engine_name
        0,                            # engine_version
        xr.XR_CURRENT_API_VERSION,    # api_version
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

    # Request OpenGL 4.0 core profile
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
        b"OpenXR Frame Loop",
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
        system_id,  # system_id (as returned by pyopenxr)
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
    # 9) Begin XR session
    # --------------------------------------------------------
    begin_info = xr.SessionBeginInfo(
        primary_view_configuration_type=view_config_type
    )
    xr.begin_session(session, begin_info)
    print("✅ Began XR session.")

    # --------------------------------------------------------
    # 10) Frame loop (no rendering, just head pose)
    # --------------------------------------------------------
    print("\n=== Entering frame loop (about 300 frames) ===")
    environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE

    frame_limit = 300
    frame_index = 0

    # Simple SDL event struct to keep the window responsive
    sdl_event = sdl2.SDL_Event()

    while frame_index < frame_limit:
        # Pump SDL events (close window to break early)
        while sdl2.SDL_PollEvent(ctypes.byref(sdl_event)) != 0:
            if sdl_event.type == sdl2.SDL_QUIT:
                frame_index = frame_limit
                break

        # Wait for frame
        frame_state = xr.wait_frame(session)
        xr.begin_frame(session)

        # 🔧 FIXED FIELD NAME HERE: view_configuration_type
        locate_info = xr.ViewLocateInfo(
            view_configuration_type=view_config_type,
            display_time=frame_state.predicted_display_time,
            space=reference_space,
        )

        view_state, located_views = xr.locate_views(session, locate_info)

        if (view_state.view_state_flags
                & xr.ViewStateFlags.POSITION_VALID_BIT
                and view_state.view_state_flags
                & xr.ViewStateFlags.ORIENTATION_VALID_BIT):
            # Use first eye as "head pose" proxy
            v0 = located_views[0]
            pos = v0.pose.position
            ori = v0.pose.orientation
            print(
                f"Frame {frame_index:4d} | "
                f"Pos: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) | "
                f"Ori: ({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f})"
            )
        else:
            print(f"Frame {frame_index:4d} | View pose not valid")

        # End frame with no layers (no rendering yet)
        frame_end_info = xr.FrameEndInfo(
            display_time=frame_state.predicted_display_time,
            environment_blend_mode=environment_blend_mode,
            layers=[],
        )
        xr.end_frame(session, frame_end_info)

        frame_index += 1
        time.sleep(0.005)

    print("\nExiting frame loop.")

    # --------------------------------------------------------
    # 11) End session & cleanup
    # --------------------------------------------------------
    xr.end_session(session)
    print("✅ Ended XR session.")

    print("Cleaning up...")
    xr.destroy_session(session)
    xr.destroy_instance(instance)
    sdl2.SDL_GL_DeleteContext(gl_context)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()
    print("✅ Clean exit.")


if __name__ == "__main__":
    main()
