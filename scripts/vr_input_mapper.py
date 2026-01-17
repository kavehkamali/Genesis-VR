from dataclasses import dataclass

from vr_xr_runtime import XrActionsState


@dataclass
class VrTeleopState:
    """
    High-level teleop semantics derived from VR controller input.

    - teleop_enabled: True while right grip is held
    - gripper: 0..1 from right trigger (0=open, 1=closed, or however you map it)
    - start_episode: pulses True for one frame on A press
    - stop_episode: pulses True for one frame on B press
    """
    teleop_enabled: bool = False
    gripper: float = 0.0
    start_episode: bool = False
    stop_episode: bool = False


class VrInputMapper:
    """
    Maps low-level XrActionsState (raw buttons/axis) to VrTeleopState.

    Semantics:
      - Grip (boolean):
          Held  -> teleop_enabled = True
          Released -> teleop_enabled = False

      - Trigger (float 0..1):
          Directly mapped to gripper in [0, 1].

      - A button:
          Rising edge (False -> True) produces start_episode=True for ONE frame.

      - B button:
          Rising edge produces stop_episode=True for ONE frame.
    """

    def __init__(self):
        # Keep track of previous button states for edge detection
        self._last_grip = False
        self._last_a = False
        self._last_b = False
        self._last_gripper = 0.0

    def update(self, xr_actions: XrActionsState) -> VrTeleopState:
        """Update teleop state from raw XR action state."""
        grip = xr_actions.grip
        trigger = xr_actions.trigger
        button_a = xr_actions.button_a
        button_b = xr_actions.button_b

        # Edge detection for A/B
        start_episode = (not self._last_a) and button_a
        stop_episode = (not self._last_b) and button_b

        teleop_enabled = grip  # simple "while held" semantics

        state = VrTeleopState(
            teleop_enabled=teleop_enabled,
            gripper=trigger,
            start_episode=start_episode,
            stop_episode=stop_episode,
        )

        # Save last values for next frame
        self._last_grip = grip
        self._last_a = button_a
        self._last_b = button_b
        self._last_gripper = trigger

        return state
