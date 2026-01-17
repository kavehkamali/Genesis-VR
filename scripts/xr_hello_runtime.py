import sys

try:
    import xr  # provided by pyopenxr
except ImportError:
    print("Error: pyopenxr is not installed. Run `pip install pyopenxr` first.")
    sys.exit(1)


def main() -> None:
    # 1) List all available OpenXR instance extensions
    print("=== Querying OpenXR extensions via pyopenxr ===")
    try:
        extensions = xr.enumerate_instance_extension_properties()
    except Exception as e:
        print("Failed to talk to the OpenXR runtime:")
        print(repr(e))
        print()
        print("Things to check:")
        print("  - Meta Quest PC app is running")
        print("  - Link / Air Link is connected")
        print("  - Meta OpenXR runtime is set as active in the PC app settings")
        sys.exit(1)

    # extensions is typically a list of strings with extension names
    print(f"Found {len(extensions)} extensions:")
    for name in extensions:
        print("  ", name)

    # 2) Check for the OpenGL extension we will need later for rendering
    required_ext = xr.KHR_OPENGL_ENABLE_EXTENSION_NAME
    print()
    print("Required extension for OpenGL rendering:")
    print("  ", required_ext)

    if required_ext in extensions:
        print("\n✅ OpenGL enable extension is available – this runtime should support GL rendering.")
        sys.exit(0)
    else:
        print("\n❌ OpenGL enable extension is NOT available.")
        print("   This usually means the active OpenXR runtime does not support OpenGL,")
        print("   or a different runtime (e.g. SteamVR) is currently active.")
        sys.exit(1)


if __name__ == "__main__":
    main()
