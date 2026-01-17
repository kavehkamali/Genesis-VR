"""
https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/keyboard_teleop.py

Keyboard Controls:
↑	- Move Forward (North)
↓	- Move Backward (South)
←	- Move Left (West)
→	- Move Right (East)
n	- Move Up
m	- Move Down
j	- Rotate Counterclockwise
k	- Rotate Clockwise
u	- Reset Scene (and save episode if enabled)
space	- Press to close gripper, release to open gripper
esc	- Quit

VR Controls (with --vr):
Grip button: engage teleoperation (move robot in space)
Trigger: close gripper (pressed) / open (released)
A: home the arm smoothly
B: save trajectory (if enabled) and reset scene (same as keyboard 'u')
"""

import os, sys, argparse, time, math
import random
import threading, uuid

import numpy as np
import genesis as gs
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
from PIL import Image
from colorama import Fore, Style
from utils import *

import openvr  # VR support

print('gs ver:', gs.__version__, gs.__file__)
assert gs.__version__ == '0.3.10'

# -------------------- VR quaternion helpers -------------------- #

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0., 0., 0., 1.], dtype=float)
    return q / n


def quat_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quat_mul(q1, q2):
    """
    Hamilton product of two quaternions (qx, qy, qz, qw).
    Result represents applying q2 then q1.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([x, y, z, w], dtype=float)


def quat_from_axis_angle(axis, angle):
    """
    axis: (3,) unit vector
    angle: radians
    returns (qx, qy, qz, qw)
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8 or abs(angle) < 1e-8:
        return np.array([0., 0., 0., 1.], dtype=float)
    axis = axis / axis_norm
    s = math.sin(angle / 2.0)
    x, y, z = axis * s
    w = math.cos(angle / 2.0)
    return np.array([x, y, z, w], dtype=float)


def quat_rotate_vec(q, v):
    """
    Rotate vector v by quaternion q (qx, qy, qz, qw).
    """
    q = quat_normalize(q)
    v = np.asarray(v, dtype=float)
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=float)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[:3]


# -------------------- OpenVR helpers -------------------- #

def get_pos_quat_from_pose(pose):
    """
    Return (pos, quat) from an OpenVR TrackedDevicePose_t.

    pos: (x, y, z)
    quat: (qx, qy, qz, qw)
    """
    m = pose.mDeviceToAbsoluteTracking

    x = m[0][3]
    y = m[1][3]
    z = m[2][3]

    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    return (x, y, z), quat_normalize([qx, qy, qz, qw])


def is_pressed(state, button_id):
    return (state.ulButtonPressed & (1 << button_id)) != 0


def choose_controller(system, poses):
    """
    Choose which controller to use:
      1) Prefer one with any button pressed,
      2) fallback to first valid controller.
    Return (idx, pose) or (None, None).
    """
    candidates = []
    for i, pose in enumerate(poses):
        if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
            continue
        if system.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_Controller:
            continue
        candidates.append((i, pose))

    if not candidates:
        return None, None

    for idx, pose in candidates:
        ok, state = system.getControllerState(idx)
        if ok and state.ulButtonPressed != 0:
            return idx, pose

    return candidates[0]


# -------------------- Argparse -------------------- #

parser = argparse.ArgumentParser()
parser.add_argument(
        "--save",
        action='store_true',
    )

parser.add_argument(
        "--save_image",
        action='store_true',
    )

parser.add_argument(
        "--vla",
        action='store_true',
    )

parser.add_argument(
        "--delta",
        action='store_true',
    )

parser.add_argument(
        "--cpu",
        action='store_true',
    )

parser.add_argument(
    "--ws",
    type=str,
    default="10.225.68.29",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=10,
)

parser.add_argument(
    "--prompt",
    type=str,
    default="lift the red cube",
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="output",
)

# New: VR mode switch
parser.add_argument(
    "--vr",
    action="store_true",
    help="Enable VR teleoperation instead of keyboard / VLA.",
)

parser.add_argument(
    "--vr-enable-rotation",
    action="store_true",
    help="Enable wrist rotation from VR controller.",
)

parser.add_argument(
    "--vr-speed",
    type=float,
    default=1.0,
    help="Speed scaling for VR translation.",
)

args = parser.parse_args()


# -------------------- Keyboard device -------------------- #

class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        try:
            self.listener.stop()
        except NotImplementedError:
            # Dummy backend does not implement stop
            pass
        self.listener.join()

    def on_press(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self):
        return self.pressed_keys


# -------------------- Scene construction -------------------- #

def build_scene():
    ########################## init ##########################
    gs.init(precision="32", logging_level="info", backend=gs.cpu if args.cpu else gs.gpu)
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
            dt=0.01  # added from kk
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            box_box_detection=True,
            constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
            camera_fov=50,
            max_FPS=20,
        ),
        show_viewer=True,
        show_FPS=False,
    )
    scene.use_shadow = False
    ########################## entities ##########################
    entities = dict()
    entities["plane"] = scene.add_entity(
        gs.morphs.Plane(),
    )

    entities["robot"] = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, 0),
        ),
    )

    cube_specs = {
        "cube110": (1, 1, 0), # yellow
        "cube001": (0, 0, 1), # blue
        "cube010": (0, 1, 0), # green
        "cube100": (1, 0, 0),  # red
        "cube000": (0, 0, 0), # black
        "cube111": (1, 1, 1), # white
        "cube101": (1, 0, 1), # purple
        "cube011": (0, 1, 1)  # cyan
    }

    size = (0.04, 0.04, 0.04)
    material = gs.materials.Rigid(rho=300)

    for cube_name, color in cube_specs.items():
        cube = scene.add_entity(
            material=material,
            morph=gs.morphs.Box(pos=(pos_rnd(1,1), pos_rnd(1,1), 0.05), size=size),
            surface=gs.surfaces.Default(color=color),
        )
        entities[cube_name] = cube

    entities["target"] = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    entities["table"] = scene.add_entity(
        gs.morphs.Box(size=(1.5, 1.5, 0.04), pos=(0.0, 0.0, 0.0), fixed=True),
        surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7, 1.0)),
    )

    # Cameras
    external_cam = scene.add_camera(res=(256, 256), pos=(1.5, 0.0, 1.0),
                                    lookat=(0.1, 0, 0.0), fov=50, GUI=False)
    wrist_cam = scene.add_camera(res=(256, 256), pos=(0.0, 0.0, 0.0),
                                lookat=(0.0, 0.0, 1.0), fov=100, GUI=False)
    varied_cam_2 = scene.add_camera(res=(256, 256), pos=(1.0, 1.0, 1.2),
                                    lookat=(0.0, 0.0, 0.0), fov=50, GUI=False)

    video_cam = scene.add_camera(res=(1280, 960), pos=(3.5, 0.0, 2.5),
                                 lookat=(0.0, 0.0, 0.5), fov=30, GUI=False,)
    ########################## build ##########################
    scene.build()

    for obj in entities.keys():
        if 'cube' in obj:
            print('before random===>', entities[obj].get_pos().cpu().numpy(), entities[obj].get_quat().cpu().numpy())
            entities[obj].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))
            print('after random===>', entities[obj].get_pos().cpu().numpy(), entities[obj].get_quat().cpu().numpy())
            print()
    scene.step()

    return scene, entities, external_cam, wrist_cam, varied_cam_2, video_cam


# -------------------- Main simulation -------------------- #

def run_sim(client_keyboard, policy):
    output_dir = os.path.abspath(args.output_dir)
    print(f"Output folder: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # only create subdirs if we’ll use them
    if args.save:
        os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)

    if args.save_image:
        os.makedirs(os.path.join(output_dir, "debug_self_collection"), exist_ok=True)

    scene, entities, external_cam, wrist_cam, varied_cam_2, video_cam = build_scene()
    robot = entities["robot"]
    target_entity = entities["target"]

    motors_dof = np.arange(robot.n_dofs - 2)
    fingers_dof = np.arange(robot.n_dofs - 2, robot.n_dofs)
    ee_link = robot.get_link("hand")

    robot_init_pos = np.array([0.5, 0, 0.55])  # domain randomization here
    robot_init_R = R.from_euler("y", np.pi)
    robot_init_quat = robot_init_R.as_quat(scalar_first=True)

    q = robot.inverse_kinematics(link=ee_link, pos=robot_init_pos, quat=robot_init_quat)
    robot.set_qpos(q[:-2], motors_dof)

    target_pos = robot_init_pos.copy()
    target_R = robot_init_R

    episode = []

    valid_step = 0

    # -------------------- VR initialization -------------------- #
    if args.vr:
        # Default workspace for VR teleop
        robot_base_pos_default = np.array([0.5, 0.0, 0.4], dtype=float)
        quat_down = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

        home_qpos = robot.inverse_kinematics(
            link=ee_link,
            pos=robot_base_pos_default,
            quat=quat_down,
        )
        if hasattr(home_qpos, "detach"):
            home_qpos = home_qpos.detach().cpu().numpy()
        else:
            home_qpos = np.array(home_qpos, dtype=float)
        home_qpos = home_qpos.copy()
        home_qpos[-2:] = 0.04  # open gripper

        last_pos_robot = robot_base_pos_default.copy()

        openvr.init(openvr.VRApplication_Background)
        system = openvr.VRSystem()

        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        poses = poses_t()

        BTN_GRIP = openvr.k_EButton_Grip
        BTN_TRIG = openvr.k_EButton_SteamVR_Trigger
        BTN_A    = openvr.k_EButton_A
        BTN_B    = openvr.k_EButton_ApplicationMenu

        base_pos_vr = None
        robot_anchor_pos = last_pos_robot.copy()

        controller_home_quat = None

        pos_forward_axis = None
        pos_right_axis = None
        pos_up_axis = np.array([0.0, 0.0, 1.0], dtype=float)

        homing = False
        homing_step = 0
        HOMING_STEPS = 150
        q_start = home_qpos.copy()

        prev_pressed_mask = 0
        frame = 0
    else:
        # Placeholders to avoid UnboundLocalError if referenced
        robot_base_pos_default = None
        quat_down = None
        last_pos_robot = None
        system = None
        poses = None
        BTN_GRIP = BTN_TRIG = BTN_A = BTN_B = None
        base_pos_vr = None
        robot_anchor_pos = None
        controller_home_quat = None
        pos_forward_axis = pos_right_axis = pos_up_axis = None
        homing = False
        homing_step = 0
        HOMING_STEPS = 0
        q_start = None
        prev_pressed_mask = 0
        frame = 0

    try:
        while 1:
            pressed_keys = client_keyboard.pressed_keys.copy()

            # allow ESC to quit regardless of mode
            for key in pressed_keys:
                if key == keyboard.Key.esc:
                    print('force quit')
                    os._exit(0)

            ee_pos = ee_link.get_pos().cpu().numpy()
            ee_quat = ee_link.get_quat().cpu().numpy()
            ee_rpy = R.from_quat(ee_quat).as_euler("xyz")
            gripper_qpos = robot.get_qpos()[fingers_dof]
            ee_state = np.concatenate([ee_pos, ee_rpy, gripper_qpos.cpu().numpy()])
            assert ee_state.shape == (8,)

            # record previous pose BEFORE key changes
            prev_pos = target_pos.copy()
            prev_R = target_R
            delta_ee = [0] * 7
            # get ee target pose
            is_close_gripper = False
            dpos = 0.002
            drot = 0.01

            ext_img = external_cam.render(rgb=True)[0].astype(np.uint8)
            look_vec = quat_rotate(ee_quat, np.array([0.0, 0.0, 1.0]))
            up_vec = quat_rotate(ee_quat, np.array([1.0, 0.0, 0.0]))
            # offset camera in local EE frame
            offset_local = np.array([0.07, 0.0, 0.0])
            wrist_pos = ee_pos + quat_rotate(ee_quat, offset_local)
            # set camera pose using quaternion
            wrist_cam.set_pose(
                pos=wrist_pos,
                lookat=wrist_pos + look_vec,
                up=up_vec
            )
            wrist_img = wrist_cam.render(rgb=True)[0].astype(np.uint8)
            if args.save_image and valid_step > 0 and valid_step % 10 == 0:
                Image.fromarray(wrist_img).save(f"{output_dir}/debug_self_collection/test_wrist_img_{valid_step}.png")

            assert ext_img.shape == (256,256,3) and wrist_img.shape == ext_img.shape
            assert ext_img.max() > 1 , f"max value of the image: {ext_img.max()}"
            assert wrist_img.max() > 1, f"max value of the image: {wrist_img.max()}"

            # -------------------- VLA policy branch -------------------- #
            if policy:  # using vla
                for key in pressed_keys:
                    if key == keyboard.Key.esc:
                        print('force quit')
                        os._exit(0)

                obs =  {
                        "observation/image":ext_img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": ee_state,
                        "prompt": args.prompt
                        }
                action_chunk = policy.infer(obs)["actions"][:args.horizon]
                assert action_chunk.shape == (args.horizon,7)
                valid_step += 1
                print('action chunk received, valid step:', valid_step)
                for action in action_chunk:
                    print('action--->', action)

                    if args.delta:
                        delta_xyz = action[0:3]
                        delta_rpy = action[3:6]
                        MAX_DELTA_POS = 0.05  # 5cm
                        MAX_DELTA_ROT = 0.1   # ~5.7deg

                        current_pos = ee_link.get_pos().cpu().numpy()
                        current_quat = ee_link.get_quat().cpu().numpy()
                        current_rot = R.from_quat(current_quat)
                        delta_rot = R.from_euler("xyz", delta_rpy)

                        target_pos = current_pos + delta_xyz
                        target_rot = current_rot * delta_rot
                        target_quat = target_rot.as_quat(scalar_first=False)
                    else:
                        target_pos = action[:3]
                        target_rpy = action[3:6]
                        roll, pitch, yaw = action[3], action[4], action[5]

                        target_quat = rpy_to_quat(roll, pitch, yaw) # 3dim --> 4dim

                    qpos, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)

                    robot.control_dofs_position(qpos[:-2], motors_dof)
                    # control gripper
                    if action[-1] > 0.2:  # close gripper
                        robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
                    else:
                        robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

                    print('-'*10)

            # -------------------- VR teleoperation branch -------------------- #
            elif args.vr:
                frame += 1

                system.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding,
                    0,
                    poses,
                )

                idx, pose = choose_controller(system, poses)
                if idx is None:
                    if frame % 60 == 0:
                        print("[VR] No VR controller tracked")
                else:
                    pos_vr, quat_vr = get_pos_quat_from_pose(pose)
                    ok, state = system.getControllerState(idx)

                    pressed_mask    = state.ulButtonPressed if ok else 0
                    grip_pressed    = is_pressed(state, BTN_GRIP) if ok else False
                    trigger_pressed = is_pressed(state, BTN_TRIG) if ok else False
                    a_pressed       = is_pressed(state, BTN_A)    if ok else False
                    b_pressed       = is_pressed(state, BTN_B)    if ok else False

                    a_just_pressed = a_pressed and not (prev_pressed_mask & (1 << BTN_A))
                    b_just_pressed = b_pressed and not (prev_pressed_mask & (1 << BTN_B))

                    controller_active = grip_pressed

                    # --- B button → same as keyboard 'u': save episode & reset scene ---
                    if b_just_pressed:
                        if args.save and len(episode) > 20:
                            assert not args.vla
                            print('begin to save episode:', len(episode))
                            np.savez_compressed(
                                f"{output_dir}/trajectories/rm_red_cube_{str(uuid.uuid4())[:4]}_{len(episode)}",
                                data=episode
                            )
                            print('npz saved')
                        episode = []

                        print('reset scene for new episode (VR B button)')
                        for obj in entities.keys():
                            if 'cube' in obj:
                                print('before random===>', entities[obj].get_pos().cpu().numpy(), entities[obj].get_quat().cpu().numpy())
                                entities[obj].set_pos((pos_rnd(1,1), pos_rnd(1,1), 0.05))
                                entities[obj].set_quat(
                                    R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True)
                                )
                                print('after random===>', entities[obj].get_pos().cpu().numpy(), entities[obj].get_quat().cpu().numpy())
                                print()

                    # --- A button → homing in joint space ---
                    if a_just_pressed:
                        q_start_raw = robot.get_qpos()
                        if hasattr(q_start_raw, "detach"):
                            q_start_raw = q_start_raw.detach().cpu().numpy()
                        else:
                            q_start_raw = np.array(q_start_raw, dtype=float)
                        q_start = q_start_raw.copy()

                        homing = True
                        homing_step = 0

                        base_pos_vr = None
                        last_pos_robot = robot_base_pos_default.copy()

                        print("[VR] A pressed -> start homing")

                    if homing:
                        t = homing_step / float(HOMING_STEPS)
                        if t > 1.0:
                            t = 1.0

                        q_cmd = (1.0 - t) * q_start + t * home_qpos
                        robot.control_dofs_position(q_cmd[:-2], motors_dof)
                        # open gripper during homing
                        robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

                        homing_step += 1
                        if homing_step > HOMING_STEPS:
                            homing = False
                            print("[VR] Finished homing")

                        prev_pressed_mask = pressed_mask
                    else:
                        # Normal teleop
                        if controller_active and base_pos_vr is None:
                            base_pos_vr = np.array(pos_vr, dtype=float)
                            robot_anchor_pos = last_pos_robot.copy()

                            if controller_home_quat is None:
                                controller_home_quat = quat_normalize(quat_vr)

                                # infer axes from rest pose
                                local_forward = np.array([0.0, 0.0, -1.0], dtype=float)
                                forearm_dir_world = quat_rotate_vec(controller_home_quat, local_forward)

                                up_world = np.array([0.0, 0.0, 1.0], dtype=float)

                                right_un = np.cross(up_world, forearm_dir_world)
                                norm_right = np.linalg.norm(right_un)
                                if norm_right < 1e-6:
                                    right_dir = np.array([0.0, 1.0, 0.0], dtype=float)
                                else:
                                    right_dir = right_un / norm_right

                                forward_un = np.cross(right_dir, up_world)
                                norm_forward = np.linalg.norm(forward_un)
                                if norm_forward < 1e-6:
                                    forward_dir = np.array([1.0, 0.0, 0.0], dtype=float)
                                else:
                                    forward_dir = forward_un / norm_forward

                                pos_forward_axis = forward_dir
                                pos_right_axis = right_dir
                                pos_up_axis = up_world

                        if not controller_active:
                            base_pos_vr = None

                        if controller_active and base_pos_vr is not None:
                            # ----- position -----
                            delta_vr = np.array(pos_vr, dtype=float) - base_pos_vr
                            scale = 0.7 * args.vr_speed

                            if pos_forward_axis is not None and pos_right_axis is not None:
                                d_fwd = np.dot(delta_vr, pos_forward_axis)
                                d_right = np.dot(delta_vr, pos_right_axis)
                                d_up = np.dot(delta_vr, pos_up_axis)
                                # X/Z swapped, Y mirrored:
                                #   X <- up,   Y <- -right,   Z <- forward
                                delta_robot = scale * np.array([d_up, -d_right, d_fwd], dtype=float)
                            else:
                                dx_vr, dy_vr, dz_vr = delta_vr
                                delta_robot = scale * np.array([dy_vr, -dx_vr, dz_vr], dtype=float)

                            pos_robot = robot_anchor_pos + delta_robot

                            # ----- rotation: relative to home, expressed in calibrated body basis -----
                            if (
                                args.vr_enable_rotation
                                and controller_home_quat is not None
                                and pos_forward_axis is not None
                                and pos_right_axis is not None
                            ):
                                # relative rotation from the calibrated "rest" orientation
                                q_rel = quat_mul(quat_vr, quat_conjugate(controller_home_quat))
                                q_rel = quat_normalize(q_rel)
                                qx, qy, qz, qw = q_rel

                                sin_half = math.sqrt(qx*qx + qy*qy + qz*qz)
                                if sin_half < 1e-6:
                                    # essentially no rotation
                                    quat_robot = quat_down.copy()
                                else:
                                    angle = 2.0 * math.atan2(sin_half, max(min(qw, 1.0), -1.0))
                                    axis_world = np.array([qx, qy, qz], dtype=float) / sin_half

                                    # Build body basis: columns = [forward, right, up] in world coords
                                    B = np.column_stack([
                                        pos_forward_axis,   # body X: along forearm
                                        pos_right_axis,     # body Y: sideways
                                        pos_up_axis,        # body Z: up
                                    ])  # shape (3,3)

                                    # Express rotation axis in body coordinates
                                    axis_body = B.T @ axis_world
                                    a_fwd, a_right, a_up = axis_body  # components around (forward, right, up)

                                    # Map body rotations -> robot world axes:
                                    KX, KY, KZ = 1.0, 1.0, 1.0  # per-axis gains

                                    rx = KX * angle * a_fwd
                                    ry = KY * angle * a_right
                                    rz = KZ * angle * a_up

                                    q_rx = quat_from_axis_angle([1.0, 0.0, 0.0], rx)
                                    q_ry = quat_from_axis_angle([0.0, 1.0, 0.0], ry)
                                    q_rz = quat_from_axis_angle([0.0, 0.0, 1.0], rz)

                                    # Apply Z, then Y, then X on top of down-facing pose
                                    quat_robot = quat_mul(q_rz, quat_mul(q_ry, quat_mul(q_rx, quat_down)))
                                    quat_robot = quat_normalize(quat_robot)
                            else:
                                quat_robot = quat_down

                            # IK and robot control
                            qpos, err = robot.inverse_kinematics(
                                link=ee_link,
                                pos=pos_robot,
                                quat=quat_robot,
                                return_error=True,
                            )

                            robot.control_dofs_position(qpos[:-2], motors_dof)

                            # Trigger controls gripper
                            if trigger_pressed:
                                robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
                            else:
                                robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

                            last_pos_robot = pos_robot.copy()

                    prev_pressed_mask = pressed_mask

            # -------------------- Keyboard teleoperation branch -------------------- #
            else:  # using keyboard
                for key in pressed_keys:
                    if key == keyboard.Key.up:
                        target_pos[0] -= dpos
                    elif key == keyboard.Key.down:
                        target_pos[0] += dpos
                    elif key == keyboard.Key.right:
                        target_pos[1] += dpos
                    elif key == keyboard.Key.left:
                        target_pos[1] -= dpos
                    elif key == keyboard.KeyCode.from_char("n"):
                        target_pos[2] += dpos
                    elif key == keyboard.KeyCode.from_char("m"):
                        target_pos[2] -= dpos
                    elif key == keyboard.KeyCode.from_char("j"):
                        target_R = R.from_euler("z", drot) * target_R
                    elif key == keyboard.KeyCode.from_char("k"):
                        target_R = R.from_euler("z", -drot) * target_R
                    elif key == keyboard.Key.space:
                        is_close_gripper = True  # close true, otherwise open as false
                    elif key == keyboard.Key.esc:
                        print('force quit')
                        os._exit(0)
                    elif key == keyboard.KeyCode.from_char("u"):
                        if args.save and len(episode) > 20:
                            assert not args.vla
                            print('begin to save episode:', len(episode))
                            np.savez_compressed(f"{output_dir}/trajectories/rm_red_cube_{str(uuid.uuid4())[:4]}_{len(episode)}", data=episode)
                            print('npz saved')
                        episode = []

                        # reset for randomization
                        print('reset scene for new episode')
                        for obj in entities.keys():
                            if 'cube' in obj:
                                print('before random===>', entities[obj].get_pos().cpu().numpy(), entities[obj].get_quat().cpu().numpy())
                                entities[obj].set_pos((pos_rnd(1,1), pos_rnd(1,1), 0.05))
                                entities[obj].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))
                                print('after random===>', entities[obj].get_pos().cpu().numpy(), entities[obj].get_quat().cpu().numpy())
                                print()

                # --- compute deltas ---
                delta_pos = target_pos - prev_pos
                delta_rpy = (target_R * prev_R.inv()).as_euler("xyz")

                if sum(delta_pos) != 0 or sum(delta_rpy) != 0 or is_close_gripper:
                    print('valid_step:', valid_step)
                    print("Δpos (x,y,z):", delta_pos.shape,  delta_pos)
                    print("Δrpy (roll,pitch,yaw):", delta_rpy.shape, delta_rpy)

                    action_delta_ee = np.concatenate([delta_pos.ravel(), delta_rpy.ravel(), np.array(int(is_close_gripper)).ravel()])
                    print('action_delta_ee concated:', action_delta_ee)
                    assert action_delta_ee.shape == (7,)

                    print('ee state:', ee_pos, ee_rpy, gripper_qpos.cpu().numpy())

                    target_quat = target_R.as_quat(scalar_first=True)
                    target_rpy = quat_to_rpy(target_quat)
                    target_quat_ = rpy_to_quat(target_rpy[0], target_rpy[1], target_rpy[2])
                    print("target_pos:", target_pos.dtype, target_pos.shape, target_pos)
                    print("target_quat:", target_quat.dtype, target_quat.shape, target_quat)
                    print("target_quat_:", target_quat_.dtype, target_quat_.shape, target_quat_)
                    print('target_quat conver diff:', target_quat_-target_quat)
                    print("target_rpy:", target_rpy.dtype, target_rpy.shape, target_rpy)
                    print('is_close_gripper:', int(is_close_gripper))

                    action_abs_ee = np.concatenate([target_pos.ravel(),  target_rpy.ravel(),  np.array(int(is_close_gripper)).ravel()])
                    assert action_abs_ee.shape == (7,)
                    print('action_abs_ee:', action_abs_ee)
                    print('-'*5+'\n')

                    valid_step += 1
                    episode.append(
                        {
                            "image": ext_img,
                            "wrist_image": wrist_img,
                            "state": ee_state.astype(np.float32),
                            "actions_delta_ee": action_delta_ee.astype(np.float32),
                            "actions": action_abs_ee.astype(np.float32),
                            "task": args.prompt
                        }
                    )

                # control arm
                target_quat = target_R.as_quat(scalar_first=True)
                target_entity.set_qpos(np.concatenate([target_pos, target_quat]))

                qpos, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)

                assert qpos.shape == (9,)
                if sum(delta_pos) != 0 or sum(delta_rpy) != 0 or is_close_gripper:
                    print('err of ik:', err.cpu().numpy().sum())

                robot.control_dofs_position(qpos[:-2], motors_dof)
                # control gripper
                if is_close_gripper:
                    robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
                else:
                    robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

                # Only keyboard branch sets `err` – error checks below will be gated.

            # -------------------- IK error checking (keyboard-only) -------------------- #
            if not args.vla and not args.vr:
                POSITION_THRESHOLD = 5e-3 # meters
                ORIENTATION_THRESHOLD = 5e-3 # radians

                pos_err = err[:3]
                ori_err = err[3:]
                try:
                    assert np.linalg.norm(pos_err.cpu().numpy()) <= POSITION_THRESHOLD, f"pos ik error:{np.linalg.norm(pos_err.cpu().numpy())}"
                except:
                    print(Fore.RED + f"Warning: pos ik error too high: {np.linalg.norm(pos_err.cpu().numpy())}" + Style.RESET_ALL)
                try:
                    assert np.linalg.norm(ori_err.cpu().numpy()) <= ORIENTATION_THRESHOLD, f"ori ik error:{np.linalg.norm(ori_err.cpu().numpy())}"
                except:
                    print(Fore.RED + f"Warning: ori ik error too high: {np.linalg.norm(ori_err.cpu().numpy())}" + Style.RESET_ALL)

            scene.step()

    finally:
        if args.vr:
            openvr.shutdown()


def main():

    client_keyboard = KeyboardDevice()
    client_keyboard.start()

    if args.vla:
        from openpi_client import websocket_client_policy
        policy = websocket_client_policy.WebsocketClientPolicy(
            host= args.ws,
            port=8000,
            api_key=None,
        )
        args.save=False
    else:
        policy = None

    run_sim(client_keyboard, policy)


if __name__ == "__main__":
    main()