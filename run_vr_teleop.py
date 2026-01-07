"""
run_vr_teleop.py

Step: map VR controller input -> teleop semantics.

Uses:
  - XrRuntime (from vr_xr_runtime.py) for OpenXR + GL + swapchains.
  - VrInputMapper (from vr_input_mapper.py) to map:
        Grip     -> teleop_enabled
        Trigger  -> gripper [0..1]
        Button A -> start_episode pulse
        Button B -> stop_episode pulse

For now, it just prints teleop events.
Later, you'll replace the print()s with calls into your robot / Genesis code.
"""

from vr_xr_runtime import XrRuntime, XrActionsState
from vr_input_mapper import VrInputMapper, VrTeleopState


def teleop_per_frame(actions: XrActionsState):
    """
    Per-frame callback: called once per XR frame with raw actions.
    We map them to VrTeleopState and log teleop-level events.
    """
    # Static-like storage for previous teleop state
    if not hasattr(teleop_per_frame, "_mapper"):
        teleop_per_frame._mapper = VrInputMapper()
        teleop_per_frame._last_state = VrTeleopState()

    mapper: VrInputMapper = teleop_per_frame._mapper
    last_state: VrTeleopState = teleop_per_frame._last_state

    # Map raw XR actions to teleop semantics
    state = mapper.update(actions)

    # --- Teleop-level logging ---

    # Teleop enabled/disabled
    if state.teleop_enabled != last_state.teleop_enabled:
        print(
            f"[TELEOP] Teleop {'ENABLED' if state.teleop_enabled else 'DISABLED'} "
            f"(grip={int(actions.grip)})"
        )

    # Gripper value (only log significant changes to avoid spam)
    if abs(state.gripper - last_state.gripper) > 0.1:
        print(f"[TELEOP] Gripper value: {state.gripper:.2f}")

    # Episode start/stop pulses
    if state.start_episode:
        print("[TELEOP] START episode (Button A pressed)")

    if state.stop_episode:
        print("[TELEOP] STOP/SAVE/RESET episode (Button B pressed)")

    # Optionally still log raw input (comment out if you don't want this)
    if actions.grip != getattr(teleop_per_frame, "_last_grip_raw", False):
        print(f"[INPUT] Grip -> {int(actions.grip)}")
    if abs(actions.trigger - getattr(teleop_per_frame, "_last_trigger_raw", 0.0)) > 0.02:
        print(f"[INPUT] Trigger -> {actions.trigger:.2f}")
    if actions.button_a != getattr(teleop_per_frame, "_last_a_raw", False):
        print(f"[INPUT] Button A -> {int(actions.button_a)}")
    if actions.button_b != getattr(teleop_per_frame, "_last_b_raw", False):
        print(f"[INPUT] Button B -> {int(actions.button_b)}")

    teleop_per_frame._last_grip_raw = actions.grip
    teleop_per_frame._last_trigger_raw = actions.trigger
    teleop_per_frame._last_a_raw = actions.button_a
    teleop_per_frame._last_b_raw = actions.button_b

    # Save teleop state for next frame
    teleop_per_frame._last_state = state


def main():
    runtime = XrRuntime(window_title="VR Teleop")
    try:
        # per_eye=None: keep blue/red demo rendering for now
        runtime.run(per_frame=teleop_per_frame, per_eye=None)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
