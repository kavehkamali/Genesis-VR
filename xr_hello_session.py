"""
xr_hello_session.py

Step 3 in our OpenXR + Python journey.

What it does:
- Creates an OpenGL 4.x core context in a tiny SDL2 window (Windows).
- Creates an OpenXR instance with XR_KHR_opengl_enable.
- Gets the HMD system_id and recommended view sizes.
- Binds the current OpenGL context to OpenXR using GraphicsBindingOpenGLWin32KHR.
- Creates an OpenXR session.
- Cleans up and exits.

No rendering, no controllers yet – just proving we can open an XR session.
"""

import ctypes
import os
import platform
import sys

import xr  # pyopenxr

# SDL2 / OpenGL
if platform.system() == "Windows":
    SDL2_DIR = r"C:\SDL2"
    if os.path.isdir(SDL2_DIR):
        os.add_dll_directory(SDL2_DIR)

import sdl2
from OpenGL import WGL        # Windows-specific WGL bindings
from OpenGL import GL         # For glGetString(GL_VERSION)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    if platform.system() != "Windows":
        print("This example is written for Windows (OpenGL + WGL).")
        sys.exit(1)

    print("=== OpenXR hello_session (OpenGL + SDL2 + pyopenxr) ===")

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

    # Only request the OpenGL enable extension (keep it simple)
    requested_extensions = [required_ext]

    # --------------------------------------------------------
    # 2) Create OpenXR instance
    # --------------------------------------------------------
    app_info = xr.ApplicationInfo(
        "pyopenxr_hello_session",  # application_name
        1,                         # application_version
        "pyopenxr",                # engine_name
        0,                         # engine_version
        xr.XR_CURRENT_API_VERSION, # api_version
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
    # 3) Get system + view config (keep system_id as returned)
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

    # Request OpenGL 4.0 core profile (meets min requirements 4.0.0)
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
        b"OpenXR Hello Session",
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

    # Print the actual GL version to confirm it's >= 4.0
    gl_version = to_str(GL.glGetString(GL.GL_VERSION))
    print("✅ Created SDL2 window + OpenGL context.")
    print("OpenGL version reported by driver:", gl_version)

    # --------------------------------------------------------
    # 6) Bind OpenGL context to OpenXR
    # --------------------------------------------------------
    graphics_binding = xr.GraphicsBindingOpenGLWin32KHR()
    graphics_binding.h_dc = WGL.wglGetCurrentDC()
    graphics_binding.h_glrc = WGL.wglGetCurrentContext()

    print("WGL current DC :", graphics_binding.h_dc)
    print("WGL current HGLRC:", graphics_binding.h_glrc)

    if not graphics_binding.h_dc or not graphics_binding.h_glrc:
        print("❌ Could not get current HDC or HGLRC from WGL.")
        # Cleanup
        sdl2.SDL_GL_DeleteContext(gl_context)
        sdl2.SDL_DestroyWindow(window)
        sdl2.SDL_Quit()
        xr.destroy_instance(instance)
        sys.exit(1)

    print("\n✅ Filled GraphicsBindingOpenGLWin32KHR with current GL context.")

    gb_ptr = ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p)

    # EXACT pattern from known-good pyopenxr + OpenGL examples:
    sci = xr.SessionCreateInfo(
        0,          # create_flags
        system_id,  # system_id (whatever pyopenxr returned)
        next=gb_ptr # graphics binding
    )

    # --------------------------------------------------------
    # 7) Create session
    # --------------------------------------------------------
    try:
        session = xr.create_session(instance, sci)
    except OSError as e:
        print("\n❌ xr.create_session failed:", repr(e))
        print("   OpenGL version was:", gl_version)
        print("   This often means the GL context does not meet requirements")
        print("   or the runtime driver rejected this WGL context.")
        # Cleanup
        sdl2.SDL_GL_DeleteContext(gl_context)
        sdl2.SDL_DestroyWindow(window)
        sdl2.SDL_Quit()
        xr.destroy_instance(instance)
        sys.exit(1)

    print("✅ Created OpenXR session:", session)

    # List reference spaces just to prove session is valid
    ref_spaces = xr.enumerate_reference_spaces(session)
    print("\nSession supports reference spaces:")
    for rs in ref_spaces:
        print("   ", xr.ReferenceSpaceType(rs))

    print("\n🎉 All good: Instance + System + GL context + Session created.")

    # --------------------------------------------------------
    # 8) Cleanup
    # --------------------------------------------------------
    print("\nCleaning up...")
    xr.destroy_session(session)
    xr.destroy_instance(instance)
    sdl2.SDL_GL_DeleteContext(gl_context)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()
    print("✅ Clean exit.")


if __name__ == "__main__":
    main()
