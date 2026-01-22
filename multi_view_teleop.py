"""
run_vr_teleop_genesis_cube_torch_multiview.py

Two-phase:
  Phase 1 (Calibration, before Genesis):
    - If vr_calibration.json exists, ask to reuse or recalibrate (BEFORE any [CALIB] output)
    - Capture CENTER, RIGHT, LEFT, UP, DOWN
    - Auto-exit calibration after completion (so Genesis starts)

  Phase 2 (Genesis teleop + MULTI-VIEW cameras):
    - Apply calibration: delta_sim = R @ delta_raw * scale
    - Cube moves ONLY while grab button is HELD
    - When button is NOT held, cube is released and physics takes over

Multi-view approach (Option 1):
  - Add multiple scene cameras with GUI=True
  - Call cam.render() each frame to update each camera's OpenCV window
"""

import os
import json
import time
import math
from dataclasses import dataclass

import torch
import genesis as gs

# Optional: helps OpenCV camera windows refresh smoothly
try:
    import cv2
except Exception:
    cv2 = None

from vr_xr_runtime import XrRuntime, XrActionsState
from vr_input_mapper import VrInputMapper, VrTeleopState


DEBUG_TELEOP = True
CALIB_PATH = "vr_calibration.json"


def _torch_vec3(x, device):
    return torch.tensor(list(x), device=device, dtype=torch.float32)


def _safe_preview(v, max_len=180):
    try:
        s = str(v)
        return s if len(s) <= max_len else s[:max_len] + "..."
    except Exception:
        return "<unprintable>"


def debug_dump_actions(actions, heading="[DEBUG] XrActionsState dump"):
    print(f"\n{heading}")
    names = [n for n in dir(actions) if not n.startswith("_")]
    print("[DEBUG] attr count:", len(names))

    poseish = [
        n for n in names
        if any(k in n.lower() for k in ("pose", "pos", "aim", "grip", "hand", "right", "left", "transform", "mat", "space", "t"))
    ]
    print("[DEBUG] pose-ish attrs (first 80):", poseish[:80])

    def show_attr(name):
        if not hasattr(actions, name):
            return
        v = getattr(actions, name)
        if v is None:
            return
        print(f"[DEBUG] actions.{name} = {_safe_preview(v)}")

    candidates = [
        "right_pos", "left_pos", "controller_pos", "hand_pos", "pose_pos", "aim_pos", "grip_pos",
        "right_pose", "left_pose", "pose", "hand_pose", "controller_pose", "aim_pose", "grip_pose",
        "right_T", "left_T", "hand_T", "pose_T", "aim_T", "grip_T",
        "right_space", "left_space", "hand_space", "aim_space", "grip_space",
    ]
    for c in candidates:
        show_attr(c)

    for c in ("trigger", "squeeze", "grip", "button_a", "button_b", "teleop_enabled"):
        show_attr(c)

    print("[DEBUG] end dump\n")


def _extract_controller_pos(actions: XrActionsState, debug: bool = False):
    # 1) direct vec3 fields
    direct_fields = (
        "right_pos",
        "right_hand_pos",
        "controller_pos",
        "hand_pos",
        "pose_pos",
        "aim_pos",
        "grip_pos",
    )
    for name in direct_fields:
        if hasattr(actions, name):
            v = getattr(actions, name)
            if v is not None:
                if debug:
                    print(f"[POSE] Using direct field '{name}' = {_safe_preview(v)}")
                return v

    # 2) nested pose objects
    pose_fields = (
        "right_pose",
        "controller_pose",
        "hand_pose",
        "pose",
        "aim_pose",
        "grip_pose",
    )
    pos_names = ("pos", "position", "translation", "p")
    for pose_name in pose_fields:
        if hasattr(actions, pose_name):
            pose = getattr(actions, pose_name)
            if pose is None:
                continue
            for pos_name in pos_names:
                if hasattr(pose, pos_name):
                    v = getattr(pose, pos_name)
                    if v is not None:
                        if debug:
                            print(f"[POSE] Using nested '{pose_name}.{pos_name}' = {_safe_preview(v)}")
                        return v

    # 3) 4x4 transform matrix
    mat_fields = ("right_T", "controller_T", "hand_T", "pose_T", "aim_T", "grip_T")
    for mat_name in mat_fields:
        if hasattr(actions, mat_name):
            T = getattr(actions, mat_name)
            if T is None:
                continue
            try:
                xyz = (float(T[0][3]), float(T[1][3]), float(T[2][3]))
                if debug:
                    print(f"[POSE] Using transform '{mat_name}' -> xyz={xyz}")
                return xyz
            except Exception:
                pass

    if debug:
        print("[POSE] NO POSE FOUND in known fields.")
    return None


# -----------------------------
# Calibration helpers
# -----------------------------

def _norm(v):
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n < 1e-9:
        return (0.0, 0.0, 0.0), 0.0
    return (v[0]/n, v[1]/n, v[2]/n), n


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _mul(a, s):
    return (a[0]*s, a[1]*s, a[2]*s)


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def _orthonormalize(x, z):
    # Gram-Schmidt: make z orthogonal to x, then normalize both
    x_u, xn = _norm(x)
    if xn < 1e-6:
        return None
    proj = _dot(z, x_u)
    z_ortho = _sub(z, _mul(x_u, proj))
    z_u, zn = _norm(z_ortho)
    if zn < 1e-6:
        return None
    y_u = _cross(z_u, x_u)  # right-handed
    y_u, yn = _norm(y_u)
    if yn < 1e-6:
        return None
    return x_u, y_u, z_u


@dataclass
class CalibrationResult:
    R_row_major: list  # 9 floats
    scale: float = 1.0


class VrCalibrator:
    """
    Runs XrRuntime and captures controller positions.
    Capture: trigger > 0.8 held ~0.3s then release.
    Stages: CENTER, RIGHT, LEFT, UP, DOWN.
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.stage_names = ["CENTER", "RIGHT", "LEFT", "UP", "DOWN"]
        self.stage_i = 0
        self.captured = {}

        self._cooldown_until = 0.0
        self._acc = (0.0, 0.0, 0.0)
        self._acc_n = 0

        self.done = False
        self.result = None

        self._have_pose = False
        self._printed_stage = -1
        self._last_no_pose_print = 0.0
        self._exit_at = None

    def _stage_prompt(self):
        s = self.stage_names[self.stage_i]
        if s == "CENTER":
            return "Hold controller at CENTER, then PRESS & HOLD TRIGGER to capture."
        if s == "RIGHT":
            return "Move controller to your PHYSICAL RIGHT, then PRESS & HOLD TRIGGER to capture."
        if s == "LEFT":
            return "Move controller to your PHYSICAL LEFT, then PRESS & HOLD TRIGGER to capture."
        if s == "UP":
            return "Move controller UP, then PRESS & HOLD TRIGGER to capture."
        if s == "DOWN":
            return "Move controller DOWN, then PRESS & HOLD TRIGGER to capture."
        return f"Stage {s}: press trigger to capture."

    def _finalize(self):
        right = self.captured["RIGHT"]
        left = self.captured["LEFT"]
        up = self.captured["UP"]
        down = self.captured["DOWN"]

        # YOUR REQUESTED AXES:
        #   physical RIGHT -> +Y
        #   physical UP    -> +Z
        raw_y = _sub(right, left)
        raw_z = _sub(up, down)

        basis = _orthonormalize(raw_y, raw_z)
        if basis is None:
            raise RuntimeError("Calibration failed: movements too small or vectors nearly collinear.")

        # basis returns (x_u, y_u, z_u) where:
        #   x_u = normalize(first arg)  => raw_y_u (RIGHT)
        #   z_u = normalize(second arg) => raw_z_u (UP)
        #   y_u = cross(z_u, x_u)       => inferred forward/back
        raw_y_u, raw_x_u, raw_z_u = basis  # reorder to match sim axes X,Y,Z

        # delta_sim = R @ delta_raw where rows of R are sim basis in raw coords
        R = [
            raw_x_u[0], raw_x_u[1], raw_x_u[2],  # sim +X (inferred fwd)
            raw_y_u[0], raw_y_u[1], raw_y_u[2],  # sim +Y (physical right)
            raw_z_u[0], raw_z_u[1], raw_z_u[2],  # sim +Z (physical up)
        ]

        self.result = CalibrationResult(R_row_major=R, scale=1.0)
        self.done = True
        self._exit_at = time.time() + 0.6

        print("\n[CALIB] Calibration complete.")
        print("[CALIB] raw_y_u (physical RIGHT -> +Y) =", raw_y_u)
        print("[CALIB] raw_x_u (inferred FWD  -> +X)  =", raw_x_u)
        print("[CALIB] raw_z_u (physical UP   -> +Z)  =", raw_z_u)
        print("[CALIB] Mapping: delta_sim = R @ delta_raw\n")
        print("[CALIB] Auto-starting Genesis...\n")

    def per_frame(self, actions: XrActionsState):
        now = time.time()
        if self.done:
            return False if (self._exit_at is not None and now >= self._exit_at) else True

        cpos = _extract_controller_pos(actions, debug=False)
        if cpos is None:
            if self.debug and (now - self._last_no_pose_print) > 2.0:
                print("[CALIB] Waiting for controller pose... (turn on controller / ensure tracking)")
                self._last_no_pose_print = now
            return True

        if not self._have_pose:
            self._have_pose = True
            print("\n==============================")
            print(" VR CALIBRATION")
            print("==============================")
            print("Capture by PRESS & HOLD TRIGGER (>0.8) ~0.3s then release.\n")

        if self.stage_i != self._printed_stage:
            self._printed_stage = self.stage_i
            print(f"\n[CALIB] Stage {self.stage_i+1}/{len(self.stage_names)}: {self.stage_names[self.stage_i]}")
            print("[CALIB]", self._stage_prompt())

        trig = float(getattr(actions, "trigger", 0.0))
        pressed = trig > 0.8

        if now < self._cooldown_until:
            return True

        if pressed:
            self._acc = _add(self._acc, (float(cpos[0]), float(cpos[1]), float(cpos[2])))
            self._acc_n += 1
        else:
            if self._acc_n > 0:
                avg = _mul(self._acc, 1.0 / float(self._acc_n))
                name = self.stage_names[self.stage_i]
                self.captured[name] = avg
                print(f"[CALIB] Captured {name} = {avg} (samples={self._acc_n})")

                self._acc = (0.0, 0.0, 0.0)
                self._acc_n = 0
                self._cooldown_until = now + 0.35

                self.stage_i += 1
                if self.stage_i >= len(self.stage_names):
                    self._finalize()

        return True


def save_calibration(path: str, calib: CalibrationResult):
    data = {"version": 1, "R_row_major": calib.R_row_major, "scale": calib.scale}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[CALIB] Wrote calibration to: {os.path.abspath(path)}")


def load_calibration(path: str) -> CalibrationResult:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    R = data["R_row_major"]
    scale = float(data.get("scale", 1.0))
    if not (isinstance(R, list) and len(R) == 9):
        raise ValueError("Invalid calibration file: R_row_major must be 9 floats.")
    return CalibrationResult(R_row_major=R, scale=scale)


def run_calibration(debug=False) -> CalibrationResult:
    calibrator = VrCalibrator(debug=debug)
    runtime = XrRuntime(window_title="VR Calibration (Trigger to Capture)")
    try:
        runtime.run(per_frame=calibrator.per_frame, per_eye=None)
    finally:
        runtime.shutdown()

    if not calibrator.done or calibrator.result is None:
        raise RuntimeError("Calibration did not complete.")
    return calibrator.result


# -----------------------------
# Teleop (Genesis + Multi-view cameras)
# -----------------------------

class GenesisCubeTeleop:
    def __init__(self, show_viewer: bool = True, debug: bool = True, calib: CalibrationResult | None = None):
        self.debug = debug
        self.calib = calib

        gs.init(backend=gs.gpu, precision="32")
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, -1.5, 1.2),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=35,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=show_viewer,
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.cube = self.scene.add_entity(gs.morphs.Box(size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.6)))

        # -------------------------
        # MULTI-VIEW CAMERAS (Option 1)
        # -------------------------
        # Each of these opens its own OpenCV window because GUI=True.
        # You must call cam.render() each frame to refresh them.
        self.cam_side = self.scene.add_camera(
            res=(640, 360),
            pos=(2.0, -1.5, 1.2),
            lookat=(0.0, 0.0, 0.25),
            fov=40,
            GUI=True,
        )

        self.cam_top = self.scene.add_camera(
            res=(640, 360),
            pos=(0.0, 0.0, 2.5),
            lookat=(0.0, 0.0, 0.25),
            fov=50,
            GUI=True,
        )

        self.scene.build()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mapper = VrInputMapper()
        self.last_state = VrTeleopState()

        self.grabbed = False
        self.hand_grab_raw = None
        self.cube_grab_pos = None

        self.kp = 200.0
        self.kd = 20.0
        self.max_force = 200.0

        self.fallback_target = torch.tensor([0.0, 0.0, 0.2], device=self.device, dtype=torch.float32)

        if hasattr(self.cube, "apply_force"):
            self._force_mode = "apply_force"
        elif hasattr(self.cube, "add_force"):
            self._force_mode = "add_force"
        else:
            self._force_mode = "set_pos"

        self._printed_once = False
        self._frame_i = 0

        if self.calib is not None:
            self.R = torch.tensor(self.calib.R_row_major, device=self.device, dtype=torch.float32).view(3, 3)
            self.scale = float(self.calib.scale)
        else:
            self.R = None
            self.scale = 1.0

        if self.debug:
            print(f"[INIT] device={self.device} force_mode={self._force_mode} calib={'YES' if self.calib else 'NO'}")
            if self.calib:
                print("[INIT] R_row_major =", self.calib.R_row_major)
                print("[INIT] scale       =", self.calib.scale)

            if cv2 is None:
                print("[WARN] cv2 not importable in this environment.")
                print("       Cameras with GUI=True rely on OpenCV windows.")
                print("       Install OpenCV (pip install opencv-python) if needed.")

    def _get_pos_vel_torch(self):
        if hasattr(self.cube, "get_pos"):
            pos = self.cube.get_pos()
        elif hasattr(self.cube, "get_position"):
            pos = self.cube.get_position()
        else:
            raise AttributeError("Cube entity has no get_pos/get_position method.")

        if hasattr(self.cube, "get_vel"):
            vel = self.cube.get_vel()
        elif hasattr(self.cube, "get_velocity"):
            vel = self.cube.get_velocity()
        else:
            vel = torch.zeros(3, device=self.device, dtype=torch.float32)

        if not torch.is_tensor(pos):
            pos = _torch_vec3(pos, self.device)
        else:
            pos = pos.to(device=self.device, dtype=torch.float32)

        if not torch.is_tensor(vel):
            vel = _torch_vec3(vel, self.device)
        else:
            vel = vel.to(device=self.device, dtype=torch.float32)

        return pos, vel

    def _raw_to_sim_delta(self, raw_delta_t: torch.Tensor) -> torch.Tensor:
        if self.R is None:
            return raw_delta_t
        return (self.R @ raw_delta_t) * self.scale

    def _compute_target_torch(self, actions: XrActionsState):
        cpos = _extract_controller_pos(actions, debug=False)

        if cpos is not None and self.hand_grab_raw is not None and self.cube_grab_pos is not None:
            hand_now_raw = _torch_vec3(cpos, self.device)
            raw_delta = hand_now_raw - self.hand_grab_raw
            sim_delta = self._raw_to_sim_delta(raw_delta)
            return self.cube_grab_pos + sim_delta

        # fallback nudges (ONLY if pose is missing while grabbed)
        step = 0.01
        dx = float(getattr(actions, "trigger", 0.0)) * step
        dy = (1.0 if getattr(actions, "button_a", False) else 0.0) * step
        dy -= (1.0 if getattr(actions, "button_b", False) else 0.0) * step
        self.fallback_target = self.fallback_target + torch.tensor([dx, dy, 0.0], device=self.device)
        self.fallback_target[2] = torch.clamp(self.fallback_target[2], min=0.05)
        return self.fallback_target.clone()

    def _apply_force_or_fallback(self, force_t: torch.Tensor, target_t: torch.Tensor):
        force_t = torch.clamp(force_t, -self.max_force, self.max_force)

        if self._force_mode in ("apply_force", "add_force"):
            fx, fy, fz = float(force_t[0]), float(force_t[1]), float(force_t[2])
            getattr(self.cube, self._force_mode)((fx, fy, fz))
            return

        tx, ty, tz = float(target_t[0]), float(target_t[1]), float(target_t[2])
        if hasattr(self.cube, "set_pos"):
            self.cube.set_pos((tx, ty, tz))
        elif hasattr(self.cube, "set_position"):
            self.cube.set_position((tx, ty, tz))

        if hasattr(self.cube, "set_vel"):
            self.cube.set_vel((0.0, 0.0, 0.0))
        elif hasattr(self.cube, "set_velocity"):
            self.cube.set_velocity((0.0, 0.0, 0.0))

    def _grab_held(self, actions: XrActionsState, state: VrTeleopState) -> bool:
        """
        Returns True ONLY while the grab/side button is CONTINUOUSLY held.
        """
        for name in ("side_button", "button_side", "grip_button", "button_grip"):
            if hasattr(actions, name):
                v = getattr(actions, name)
                if isinstance(v, bool):
                    return v
                try:
                    return float(v) > 0.5
                except Exception:
                    pass

        for name in ("squeeze", "grip"):
            if hasattr(actions, name):
                try:
                    if float(getattr(actions, name)) > 0.6:
                        return True
                except Exception:
                    pass

        for name in ("grip_held", "grab_held", "teleop_held"):
            if hasattr(state, name):
                return bool(getattr(state, name))

        return False

    def _update_multiview_cameras(self):
        """
        Call render() to refresh each camera's OpenCV window (GUI=True).
        """
        try:
            self.cam_side.render()  # rgb by default
            self.cam_top.render()
            if cv2 is not None:
                # Helps avoid black/laggy OpenCV windows in some setups
                cv2.waitKey(1)
        except Exception as e:
            # Don’t crash teleop if rendering fails.
            if self.debug and (self._frame_i % 60 == 0):
                print("[WARN] camera render failed:", repr(e))

    def per_frame(self, actions: XrActionsState):
        self._frame_i += 1
        state = self.mapper.update(actions)

        if self.debug and not self._printed_once:
            self._printed_once = True
            debug_dump_actions(actions, heading="[DEBUG] FIRST FRAME ACTIONS DUMP")
            _extract_controller_pos(actions, debug=True)

        held = self._grab_held(actions, state)

        if held and not self.grabbed:
            self.grabbed = True
            cube_pos, _ = self._get_pos_vel_torch()

            if self.debug:
                debug_dump_actions(actions, heading="[DEBUG] ON GRAB ACTIONS DUMP")

            cpos = _extract_controller_pos(actions, debug=self.debug)
            self.cube_grab_pos = cube_pos.clone()

            if cpos is not None:
                self.hand_grab_raw = _torch_vec3(cpos, self.device)
                if self.debug:
                    print("[CUBE] Grab anchors:")
                    print("       cube_grab_pos =", self.cube_grab_pos.tolist())
                    print("       hand_grab_raw =", self.hand_grab_raw.tolist())
            else:
                self.hand_grab_raw = None
                self.fallback_target = cube_pos.clone()
                print("[WARN] Grabbed but NO HAND POSE found -> fallback nudges will move cube.")

            print("[CUBE] Grabbed (button held)")

        if (not held) and self.grabbed:
            self.grabbed = False
            self.hand_grab_raw = None
            self.cube_grab_pos = None
            print("[CUBE] Released (button up) - physics only")

        if self.grabbed:
            cpos = _extract_controller_pos(actions, debug=True)
            target = self._compute_target_torch(actions)
            pos, vel = self._get_pos_vel_torch()

            err = target - pos
            force = self.kp * err - self.kd * vel
            self._apply_force_or_fallback(force, target)

            if self.debug and (self._frame_i % 20 == 0):
                if cpos is None:
                    print("[DEBUG] grabbed: NO POSE FOUND -> using fallback target")
                else:
                    print(f"[DEBUG] hand_now_raw={_safe_preview(cpos)}")
                print(f"[DEBUG] pos={pos.tolist()} target={target.tolist()} vel={vel.tolist()} err={err.tolist()}")

        self.scene.step()

        # Update multi-view camera windows AFTER stepping the sim
        self._update_multiview_cameras()

        self.last_state = state


# -----------------------------
# Main
# -----------------------------

def main():
    calib = None

    # Ask about using calibration BEFORE any calibration runtime starts
    if os.path.exists(CALIB_PATH):
        print(f"[BOOT] Found calibration file: {os.path.abspath(CALIB_PATH)}")
        ans = input("Use existing calibration? [Enter=yes / c=recalibrate] : ").strip().lower()
        if ans == "c":
            calib = run_calibration(debug=DEBUG_TELEOP)
            save_calibration(CALIB_PATH, calib)
        else:
            calib = load_calibration(CALIB_PATH)
            print("[BOOT] Loaded calibration.")
    else:
        print("[BOOT] No calibration file found. Starting calibration...")
        calib = run_calibration(debug=DEBUG_TELEOP)
        save_calibration(CALIB_PATH, calib)

    teleop = GenesisCubeTeleop(show_viewer=True, debug=DEBUG_TELEOP, calib=calib)
    runtime = XrRuntime(window_title="VR Teleop + Genesis Cube (Calibrated Hand Delta) + MultiView Cameras")
    try:
        runtime.run(per_frame=teleop.per_frame, per_eye=None)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
