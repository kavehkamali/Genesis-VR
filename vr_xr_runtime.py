"""
vr_xr_runtime.py

Reusable OpenXR + SDL2 + OpenGL runtime for Meta Quest (Link) on Windows.

Adds:
- Right-hand pose publishing:
    actions.right_pos = (x,y,z)  # meters in LOCAL space
Prefers:
- aim pose (/input/aim/pose)
Fallback:
- grip pose (/input/grip/pose)

Running this file directly starts a demo:
    * blue left / red right
    * prints Grip / Trigger / Button A / Button B on changes
    * prints right_pos occasionally when valid
"""

import ctypes
import os
import platform
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import xr  # pyopenxr

# SDL2 / OpenGL
if platform.system() == "Windows":
    SDL2_DIR = r"C:\\SDL2"
    if os.path.isdir(SDL2_DIR):
        os.add_dll_directory(SDL2_DIR)

import sdl2
from OpenGL import WGL
from OpenGL import GL

DEBUG = False
def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


@dataclass
class XrActionsState:
    """Right-hand controller state we care about."""
    grip: bool = False
    trigger: float = 0.0
    button_a: bool = False
    button_b: bool = False

    # NEW: right-hand pose (LOCAL space)
    right_pos: Optional[Tuple[float, float, float]] = None
    right_pose: Optional[xr.Posef] = None  # full pose if you ever want orientation too


class XrRuntime:
    """
    Wraps:
      - OpenXR instance/system/session
      - SDL2 window + OpenGL context
      - Swapchains per eye
      - Right-hand actions: grip, trigger, A, B
      - Right-hand pose: aim pose (preferred) + grip pose (fallback)

    per_frame(actions: XrActionsState) is called once per XR frame.
    per_eye(...) is called per eye.
    """

    def __init__(self, window_title: str = "OpenXR Runtime", log_setup: bool = True):
        if platform.system() != "Windows":
            raise RuntimeError("XrRuntime is written for Windows (OpenGL + WGL).")

        self.log_setup = log_setup
        self._pose_debug = True  # keep this ON for now (non-spammy prints)

        if self.log_setup:
            print("=== XrRuntime: init ===")

        # --- 1) Extensions ---
        exts = xr.enumerate_instance_extension_properties()
        available = set()
        for e in exts:
            name = e.extension_name
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            available.add(name)

        if self.log_setup:
            print("Available extensions:")
            for n in sorted(available):
                print("   ", n)

        required_ext = xr.KHR_OPENGL_ENABLE_EXTENSION_NAME
        if isinstance(required_ext, bytes):
            required_ext = required_ext.decode("utf-8", errors="ignore")

        if self.log_setup:
            print("\nRequired extension for OpenGL rendering:")
            print("   ", required_ext)

        if required_ext not in available:
            raise RuntimeError(f"Required extension {required_ext} not available")

        if self.log_setup:
            print("\n✅ XR_KHR_opengl_enable is available.")

        # --- 2) Instance ---
        app_info = xr.ApplicationInfo(
            "pyopenxr_vr_runtime",
            1,
            "pyopenxr",
            0,
            xr.XR_CURRENT_API_VERSION,
        )

        ici = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=[required_ext],
        )

        self.instance = xr.create_instance(ici)

        if self.log_setup:
            print("\n✅ Created OpenXR instance.")
            instance_props = xr.get_instance_properties(self.instance)
            runtime_name = _to_str(instance_props.runtime_name)
            ver = instance_props.runtime_version
            print(f"Runtime name   : {runtime_name}")
            print(f"Runtime version: {ver.major}.{ver.minor}.{ver.patch}")

        # --- 3) System + views ---
        get_info = xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        self.system_id = xr.get_system(self.instance, get_info)

        if self.log_setup:
            print(f"\n✅ Got HMD system_id: {self.system_id!r} (type: {type(self.system_id).__name__})")

        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self.view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self.view_config_type
        )

        if self.log_setup:
            print(f"Number of views (eyes): {len(self.view_config_views)}")
            for i, v in enumerate(self.view_config_views):
                print(f"\nEye {i}:")
                print(f"  Recommended width : {v.recommended_image_rect_width}")
                print(f"  Recommended height: {v.recommended_image_rect_height}")
                print(f"  Max width         : {v.max_image_rect_width}")
                print(f"  Max height        : {v.max_image_rect_height}")
                print(f"  Recommended samples: {v.recommended_swapchain_sample_count}")
                print(f"  Max samples        : {v.max_swapchain_sample_count}")

        # --- 4) GL requirements ---
        pfn_get_gl_reqs = ctypes.cast(
            xr.get_instance_proc_addr(
                self.instance,
                "xrGetOpenGLGraphicsRequirementsKHR",
            ),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )

        graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        res = pfn_get_gl_reqs(self.instance, self.system_id, ctypes.byref(graphics_requirements))
        res = xr.exception.check_result(xr.Result(res))
        if res.is_exception():
            raise res

        if self.log_setup:
            print("\nOpenGL graphics requirements:")
            print("  Min API version:", graphics_requirements.min_api_version_supported)
            print("  Max API version:", graphics_requirements.max_api_version_supported)

        # --- 5) SDL2 + GL window ---
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            xr.destroy_instance(self.instance)
            raise RuntimeError("SDL initialization failed")

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 0)
        sdl2.SDL_GL_SetAttribute(
            sdl2.SDL_GL_CONTEXT_PROFILE_MASK,
            sdl2.SDL_GL_CONTEXT_PROFILE_CORE,
        )

        window_width = self.view_config_views[0].recommended_image_rect_width // 2
        window_height = self.view_config_views[0].recommended_image_rect_height // 2

        if self.log_setup:
            print(f"\nCreating SDL2 window: {window_width} x {window_height}")

        self.window = sdl2.SDL_CreateWindow(
            window_title.encode("utf-8"),
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            window_width,
            window_height,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_SHOWN,
        )
        if not self.window:
            sdl2.SDL_Quit()
            xr.destroy_instance(self.instance)
            raise RuntimeError("Failed to create SDL window")

        self.gl_context = sdl2.SDL_GL_CreateContext(self.window)
        if not self.gl_context:
            sdl2.SDL_DestroyWindow(self.window)
            sdl2.SDL_Quit()
            xr.destroy_instance(self.instance)
            raise RuntimeError("Failed to create OpenGL context")

        sdl2.SDL_GL_MakeCurrent(self.window, self.gl_context)
        sdl2.SDL_GL_SetSwapInterval(0)

        gl_version = _to_str(GL.glGetString(GL.GL_VERSION))
        if self.log_setup:
            print("✅ Created SDL2 window + OpenGL context.")
            print("OpenGL version reported by driver:", gl_version)

        # --- 6) Bind GL to XR ---
        graphics_binding = xr.GraphicsBindingOpenGLWin32KHR()
        graphics_binding.h_dc = WGL.wglGetCurrentDC()
        graphics_binding.h_glrc = WGL.wglGetCurrentContext()

        if self.log_setup:
            print("WGL current DC   :", graphics_binding.h_dc)
            print("WGL current HGLRC:", graphics_binding.h_glrc)

        if not graphics_binding.h_dc or not graphics_binding.h_glrc:
            self._cleanup_basic()
            raise RuntimeError("Could not get current HDC or HGLRC from WGL.")

        if self.log_setup:
            print("\n✅ Filled GraphicsBindingOpenGLWin32KHR with current GL context.")

        gb_ptr = ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p)
        sci = xr.SessionCreateInfo(
            0,
            self.system_id,
            next=gb_ptr
        )

        # --- 7) Session ---
        self.session = xr.create_session(self.instance, sci)
        if self.log_setup:
            print("✅ Created OpenXR session:", self.session)

        # --- 8) Reference space (LOCAL) ---
        identity_orientation = xr.Quaternionf(0.0, 0.0, 0.0, 1.0)
        identity_position = xr.Vector3f(0.0, 0.0, 0.0)
        self.identity_pose = xr.Posef(
            orientation=identity_orientation,
            position=identity_position,
        )

        ref_space_info = xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.LOCAL,
            pose_in_reference_space=self.identity_pose,
        )
        self.reference_space = xr.create_reference_space(self.session, ref_space_info)
        if self.log_setup:
            print("✅ Created LOCAL reference space:", self.reference_space)

        # --- 9) Swapchains ---
        self.color_format = self._choose_swapchain_format(self.session)

        self.swapchains = []
        self.swapchain_images = []
        self.image_sizes = []

        for i, v in enumerate(self.view_config_views):
            width = v.recommended_image_rect_width
            height = v.recommended_image_rect_height
            sample_count = v.recommended_swapchain_sample_count or 1

            sci = xr.SwapchainCreateInfo(
                create_flags=0,
                usage_flags=(
                    xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                    | xr.SwapchainUsageFlags.SAMPLED_BIT
                ),
                format=self.color_format,
                sample_count=sample_count,
                width=width,
                height=height,
                face_count=1,
                array_size=1,
                mip_count=1,
            )
            sc = xr.create_swapchain(self.session, sci)
            if self.log_setup:
                print(f"✅ Created swapchain for eye {i}: {sc}")

            images = xr.enumerate_swapchain_images(sc, xr.SwapchainImageOpenGLKHR)
            if self.log_setup:
                print(f"   Swapchain {i} has {len(images)} images.")
            self.swapchains.append(sc)
            self.swapchain_images.append(images)
            self.image_sizes.append((width, height))

        self.fbo = GL.glGenFramebuffers(1)

        # --- 10) Actions (right hand grip/trigger/A/B + POSE) ---
        if self.log_setup:
            print("\nCreating action set + actions...")

        action_set_info = xr.ActionSetCreateInfo(
            action_set_name="teleop",
            localized_action_set_name="Teleop Actions",
            priority=0,
        )
        self.action_set = xr.create_action_set(self.instance, action_set_info)

        self.right_hand_path = xr.string_to_path(self.instance, "/user/hand/right")

        # Button/axis actions
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

        # NEW: Pose actions (aim preferred, grip fallback)
        # NOTE: some pyopenxr builds name this ActionType.POSE_INPUT; if yours differs,
        # check xr.ActionType enums.
        self.right_aim_pose_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="right_aim_pose",
                action_type=xr.ActionType.POSE_INPUT,
                localized_action_name="Right Aim Pose",
                subaction_paths=[self.right_hand_path],
            ),
        )
        self.right_grip_pose_action = xr.create_action(
            self.action_set,
            xr.ActionCreateInfo(
                action_name="right_grip_pose",
                action_type=xr.ActionType.POSE_INPUT,
                localized_action_name="Right Grip Pose",
                subaction_paths=[self.right_hand_path],
            ),
        )

        # Bindings (Oculus Touch)
        oculus_touch_profile = xr.string_to_path(
            self.instance, "/interaction_profiles/oculus/touch_controller"
        )

        suggested_bindings = [
            xr.ActionSuggestedBinding(
                action=self.grip_action,
                binding=xr.string_to_path(
                    self.instance, "/user/hand/right/input/squeeze/value"
                ),
            ),
            xr.ActionSuggestedBinding(
                action=self.trigger_action,
                binding=xr.string_to_path(
                    self.instance, "/user/hand/right/input/trigger/value"
                ),
            ),
            xr.ActionSuggestedBinding(
                action=self.button_a_action,
                binding=xr.string_to_path(
                    self.instance, "/user/hand/right/input/a/click"
                ),
            ),
            xr.ActionSuggestedBinding(
                action=self.button_b_action,
                binding=xr.string_to_path(
                    self.instance, "/user/hand/right/input/b/click"
                ),
            ),

            # NEW: pose bindings
            xr.ActionSuggestedBinding(
                action=self.right_aim_pose_action,
                binding=xr.string_to_path(
                    self.instance, "/user/hand/right/input/aim/pose"
                ),
            ),
            xr.ActionSuggestedBinding(
                action=self.right_grip_pose_action,
                binding=xr.string_to_path(
                    self.instance, "/user/hand/right/input/grip/pose"
                ),
            ),
        ]

        ip_suggest_info = xr.InteractionProfileSuggestedBinding(
            interaction_profile=oculus_touch_profile,
            suggested_bindings=suggested_bindings,
        )
        xr.suggest_interaction_profile_bindings(self.instance, ip_suggest_info)
        if self.log_setup:
            print("✅ Suggested bindings for Oculus Touch (including aim/grip pose).")

        attach_info = xr.SessionActionSetsAttachInfo(action_sets=[self.action_set])
        xr.attach_session_action_sets(self.session, attach_info)
        if self.log_setup:
            print("✅ Attached action set to session.")

        # --- 11) Begin session ---
        begin_info = xr.SessionBeginInfo(primary_view_configuration_type=self.view_config_type)
        xr.begin_session(self.session, begin_info)
        if self.log_setup:
            print("✅ Began XR session.")

        self.active_action_set = xr.ActiveActionSet(
            action_set=self.action_set,
            subaction_path=xr.NULL_PATH,
        )

        # Create action spaces for pose actions (now that session exists)
        # These are the spaces we locate each frame to get controller position.
        self.right_aim_space = xr.create_action_space(
            self.session,
            xr.ActionSpaceCreateInfo(
                action=self.right_aim_pose_action,
                subaction_path=self.right_hand_path,
                pose_in_action_space=self.identity_pose,
            ),
        )
        self.right_grip_space = xr.create_action_space(
            self.session,
            xr.ActionSpaceCreateInfo(
                action=self.right_grip_pose_action,
                subaction_path=self.right_hand_path,
                pose_in_action_space=self.identity_pose,
            ),
        )
        if self.log_setup:
            print("✅ Created right-hand action spaces (aim + grip).")

        self.environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE
        self.session_running = True

        # Debug: print pose validity changes without spamming
        self._last_pose_valid = None
        self._last_pose_print_t = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _choose_swapchain_format(self, session):
        available_formats = xr.enumerate_swapchain_formats(session)
        candidates = [GL.GL_SRGB8_ALPHA8, GL.GL_RGBA8]
        for f in available_formats:
            if f in candidates:
                if self.log_setup:
                    print(f"Using swapchain format: 0x{f:08X}")
                return f
        if self.log_setup:
            print("No preferred format found, using first available:", available_formats[0])
        return available_formats[0]

    def _cleanup_basic(self):
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
        try:
            xr.destroy_instance(self.instance)
        except Exception:
            pass

    def _locate_right_pose(self, predicted_display_time, actions_state: XrActionsState):
        """
        Fill actions_state.right_pos + right_pose by locating aim space first, then grip space.
        """
        def locate(space):
            try:
                loc = xr.locate_space(space, self.reference_space, predicted_display_time)
                return loc
            except Exception:
                return None

        loc = locate(self.right_aim_space)
        used = "aim"

        # If aim isn't valid, try grip
        if loc is None or not (loc.location_flags & xr.SpaceLocationFlags.POSITION_VALID_BIT):
            loc2 = locate(self.right_grip_space)
            if loc2 is not None and (loc2.location_flags & xr.SpaceLocationFlags.POSITION_VALID_BIT):
                loc = loc2
                used = "grip"

        if loc is not None and (loc.location_flags & xr.SpaceLocationFlags.POSITION_VALID_BIT):
            p = loc.pose.position
            actions_state.right_pos = (float(p.x), float(p.y), float(p.z))
            actions_state.right_pose = loc.pose

            # Non-spammy debug: print validity toggles and occasional positions
            if self._pose_debug:
                now = time.time()
                valid = True
                if self._last_pose_valid is not True:
                    print(f"[XR POSE] right pose VALID (using {used})")
                # Print position about ~2x/sec
                if (now - self._last_pose_print_t > 0.5) and DEBUG:
                    self._last_pose_print_t = now
                    print(f"[XR POSE] {used} right_pos = {actions_state.right_pos}")
                self._last_pose_valid = valid
        else:
            actions_state.right_pos = None
            actions_state.right_pose = None
            if self._pose_debug:
                valid = False
                if self._last_pose_valid is not False:
                    print("[XR POSE] right pose NOT valid (aim+grip)")
                self._last_pose_valid = valid

    def shutdown(self):
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        try:
            if self.session_running:
                xr.end_session(self.session)
                if self.log_setup:
                    print("✅ Ended XR session.")
        except Exception as e:
            if self.log_setup:
                print("⚠ xr.end_session raised (ignored for cleanup):", repr(e))

        if self.log_setup:
            print("Cleaning up...")

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

        if self.log_setup:
            print("✅ Clean exit.")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, per_frame=None, per_eye=None):
        if not self.session_running:
            raise RuntimeError("XR session is not running")

        print("\n=== XrRuntime: entering main loop (ESC or close window to quit) ===")

        running = True
        frame_index = 0
        sdl_event = sdl2.SDL_Event()
        printed_unfocused_warning = False

        while running:
            # --- SDL events ---
            while sdl2.SDL_PollEvent(ctypes.byref(sdl_event)) != 0:
                if sdl_event.type == sdl2.SDL_QUIT:
                    running = False
                    break
                if sdl_event.type == sdl2.SDL_KEYDOWN:
                    if sdl_event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                        running = False
                        break
            if not running:
                break

            # --- XR events ---
            while True:
                try:
                    event = xr.poll_event(self.instance)
                except xr.exception.EventUnavailable:
                    break
                if isinstance(event, xr.EventDataSessionStateChanged):
                    print(f"[XR] Session state changed -> {event.state.name}")

            # --- Frame timing ---
            frame_state = xr.wait_frame(self.session)
            xr.begin_frame(self.session)

            # --- Sync actions ---
            sync_info = xr.ActionsSyncInfo(active_action_sets=[self.active_action_set])
            try:
                xr.sync_actions(self.session, sync_info)
            except xr.exception.SessionNotFocused:
                if not printed_unfocused_warning:
                    print("[XR] Warning: SessionNotFocused – app is not the focused immersive "
                          "VR app. Make sure it's launched as a full VR title in Link "
                          "(not just a desktop panel).")
                    printed_unfocused_warning = True
            except Exception as e:
                print("[XR] sync_actions error:", repr(e))

            # --- Read action states into XrActionsState ---
            actions_state = XrActionsState()
            try:
                grip_state = xr.get_action_state_boolean(
                    self.session,
                    xr.ActionStateGetInfo(action=self.grip_action, subaction_path=xr.NULL_PATH),
                )
                trigger_state = xr.get_action_state_float(
                    self.session,
                    xr.ActionStateGetInfo(action=self.trigger_action, subaction_path=xr.NULL_PATH),
                )
                a_state = xr.get_action_state_boolean(
                    self.session,
                    xr.ActionStateGetInfo(action=self.button_a_action, subaction_path=xr.NULL_PATH),
                )
                b_state = xr.get_action_state_boolean(
                    self.session,
                    xr.ActionStateGetInfo(action=self.button_b_action, subaction_path=xr.NULL_PATH),
                )

                actions_state.grip = bool(grip_state.current_state)
                actions_state.trigger = float(trigger_state.current_state)
                actions_state.button_a = bool(a_state.current_state)
                actions_state.button_b = bool(b_state.current_state)

            except Exception as e:
                print("[XR] get_action_state error:", repr(e))

            # --- NEW: locate right controller pose into actions_state.right_pos ---
            self._locate_right_pose(frame_state.predicted_display_time, actions_state)

            # --- User per-frame callback ---
            if per_frame is not None:
                keep_running = per_frame(actions_state)
                if keep_running is False:
                    # allow callers (calibration) to terminate the XR loop
                    break


            frame_index += 1
            if frame_index % 600 == 0 and DEBUG:
                print(f"[FRAME] {frame_index} frames rendered")

            # --- Views + rendering ---
            locate_info = xr.ViewLocateInfo(
                view_configuration_type=self.view_config_type,
                display_time=frame_state.predicted_display_time,
                space=self.reference_space,
            )
            view_state, located_views = xr.locate_views(self.session, locate_info)

            proj_views = []

            for eye_index, sc in enumerate(self.swapchains):
                w, h = self.image_sizes[eye_index]

                acquire_info = xr.SwapchainImageAcquireInfo()
                image_index = xr.acquire_swapchain_image(sc, acquire_info)

                wait_info = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
                xr.wait_swapchain_image(sc, wait_info)

                gl_image = self.swapchain_images[eye_index][image_index].image

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
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

                if per_eye is not None:
                    per_eye(
                        eye_index,
                        located_views[eye_index].pose,
                        located_views[eye_index].fov,
                        self.fbo,
                        w,
                        h,
                    )
                else:
                    if eye_index == 0:
                        GL.glClearColor(0.0, 0.0, 1.0, 1.0)
                    else:
                        GL.glClearColor(1.0, 0.0, 0.0, 1.0)
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT)

                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

                release_info = xr.SwapchainImageReleaseInfo()
                xr.release_swapchain_image(sc, release_info)

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

            proj_layer = xr.CompositionLayerProjection(
                layer_flags=xr.CompositionLayerFlags.NONE,
                space=self.reference_space,
                views=proj_views,
            )
            layers = [ctypes.pointer(proj_layer)]

            frame_end_info = xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=self.environment_blend_mode,
                layers=layers,
            )
            xr.end_frame(self.session, frame_end_info)

            time.sleep(0.001)

        self.session_running = False


# ----------------------------------------------------------------------
# Demo entry point
# ----------------------------------------------------------------------

def _demo_per_frame(actions: XrActionsState):
    if not hasattr(_demo_per_frame, "_last"):
        _demo_per_frame._last = XrActionsState()

    last = _demo_per_frame._last

    if actions.grip != last.grip:
        print(f"[INPUT] Grip -> {int(actions.grip)}")

    if abs(actions.trigger - last.trigger) > 0.02:
        print(f"[INPUT] Trigger -> {actions.trigger:.2f}")

    if actions.button_a != last.button_a:
        print(f"[INPUT] Button A -> {int(actions.button_a)}")

    if actions.button_b != last.button_b:
        print(f"[INPUT] Button B -> {int(actions.button_b)}")

    _demo_per_frame._last = XrActionsState(
        grip=actions.grip,
        trigger=actions.trigger,
        button_a=actions.button_a,
        button_b=actions.button_b,
        right_pos=actions.right_pos,
        right_pose=actions.right_pose,
    )


def main():
    runtime = XrRuntime(window_title="OpenXR Runtime Demo (with Pose)")
    try:
        runtime.run(per_frame=_demo_per_frame, per_eye=None)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
