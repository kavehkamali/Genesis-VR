import xr
import sys


def main():
    print("=== OpenXR hello_system with pyopenxr ===")

    # 1) Enumerate extensions (ExtensionProperties objects)
    available_props = xr.enumerate_instance_extension_properties()
    print(f"Found {len(available_props)} extensions:")

    # Extract names as Python strings
    available_names = []
    for prop in available_props:
        # Depending on pyopenxr version this may already be str or bytes
        name = prop.extension_name
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        available_names.append(name)
        print("  ", name)

    # 2) Check for OpenGL extension (we'll need it later for real rendering)
    required_extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
    print("\nRequired extension for OpenGL rendering:")
    for ext in required_extensions:
        print("  ", ext)
        if ext not in available_names:
            print(
                f"\n❌ Required extension {ext} NOT available.\n"
                "   Make sure your OpenXR runtime is set correctly (Meta Quest Link / Oculus)."
            )
            sys.exit(1)

    print("\n✅ All required extensions are available.")

    # 3) Create an OpenXR instance
    app_info = xr.ApplicationInfo(
        application_name="pyopenxr_hello_system",
        application_version=1,
        engine_name="pyopenxr",
        engine_version=1,
        api_version=xr.XR_CURRENT_API_VERSION,
    )

    ici = xr.InstanceCreateInfo(
        application_info=app_info,
        enabled_extension_names=required_extensions,
    )

    instance = xr.create_instance(ici)
    print("\n✅ Created OpenXR instance.")

    # 4) Query runtime info
    props = xr.get_instance_properties(instance)
    runtime_name = props.runtime_name
    if isinstance(runtime_name, bytes):
        runtime_name = runtime_name.decode("utf-8", errors="ignore")
    print(f"Runtime name   : {runtime_name}")
    print(f"Runtime version: {props.runtime_version}")

    # 5) Get system for HMD (your Quest via Link)
    get_info = xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
    system_id = xr.get_system(instance, get_info)
    print(f"\n✅ Got HMD system_id: {system_id}")

    # 6) Enumerate view configurations (e.g., PRIMARY_STEREO)
    view_config_types = xr.enumerate_view_configurations(instance, system_id)
    print("\nSupported view configurations:")
    for v in view_config_types:
        try:
            vtype = xr.ViewConfigurationType(v)
        except ValueError:
            vtype = v
        print("  ", vtype)

    # 7) Query the stereo view configuration info
    view_type = xr.ViewConfigurationType.PRIMARY_STEREO
    if view_type.value not in view_config_types:
        print("\n❌ PRIMARY_STEREO view configuration not supported by this runtime.")
        print("   This would be very unusual for a Quest via Link.")
        xr.destroy_instance(instance)
        sys.exit(1)

    view_config_views = xr.enumerate_view_configuration_views(
        instance,
        system_id,
        view_type,
    )

    print(f"\nView configuration: {view_type}")
    print(f"Number of views (eyes): {len(view_config_views)}")

    for i, v in enumerate(view_config_views):
        print(f"\nEye {i}:")
        print(f"  Recommended width : {v.recommended_image_rect_width}")
        print(f"  Recommended height: {v.recommended_image_rect_height}")
        print(f"  Max width         : {v.max_image_rect_width}")
        print(f"  Max height        : {v.max_image_rect_height}")
        print(f"  Recommended samples: {v.recommended_swapchain_sample_count}")
        print(f"  Max samples        : {v.max_swapchain_sample_count}")

    # 8) Clean up
    xr.destroy_instance(instance)
    print("\n✅ Destroyed instance, exiting cleanly.")
    print("\nIf you see sensible resolutions above (e.g., ~1k–2k width per eye),")
    print("your headset is correctly recognized and we’re ready for the next step.")


if __name__ == "__main__":
    main()
