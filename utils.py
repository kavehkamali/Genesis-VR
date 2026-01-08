import numpy as np
from scipy.spatial.transform import Rotation as R
import math,random


def rotate_vector(vec, quat):
    # quat = [w, x, y, z]
    u = quat[1:]
    s = quat[0]
    return 2 * np.dot(u, vec) * u + (s**2 - np.dot(u, u)) * vec + 2 * s * np.cross(u, vec)


# Kaveh's function  
def quat_to_rotvec(quat):
    w = quat[0]
    xyz = quat[1:]
    norm = np.linalg.norm(xyz)
    if norm < 1e-6:
        return np.zeros(3)
    axis = xyz / norm
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    return axis * angle

# openpi function
def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# Quaternion to axis-angle conversion (x, y, z, w)
def quat_xyzw_to_axis_angle(quat):
    """
    Convert quaternion [x, y, z, w] to axis-angle representation.
    
    Args:
        quat (array-like): Quaternion [x, y, z, w]
        
    Returns:
        axis (np.ndarray): 3D unit axis vector
        angle (float): rotation angle in radians
    """
    quat = np.array(quat, dtype=float)
    if quat.shape != (4,):
        raise ValueError("Quaternion must be a 4-element array [x, y, z, w]")
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    x, y, z, w = quat
    angle = 2 * np.arccos(w)
    s = np.sqrt(1 - w*w)  # sin(theta/2)
    if s < 1e-8:  # angle ~ 0, axis undefined
        axis = np.array([1.0, 0.0, 0.0])  # default
    else:
        axis = np.array([x, y, z]) / s
    return axis, angle


# Quaternion to axis-angle conversion (w, x, y, z)
def quat_wxyz_to_axis_angle(quat):
    """
    Convert a quaternion [w, x, y, z] to axis-angle representation.
    
    Args:
        quat: array-like of shape (4,), [w, x, y, z]
    
    Returns:
        axis: np.ndarray shape (3,), unit vector
        angle: float, rotation angle in radians
    """
    quat = np.asarray(quat, dtype=float)
    if quat.shape != (4,):
        raise ValueError("Input quaternion must be shape (4,) [w, x, y, z]")
    # Reorder to [x, y, z, w] for scipy
    q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    r = R.from_quat(q_scipy)
    rotvec = r.as_rotvec()  # returns theta * axis
    return rotvec


# Rotation vector to quaternion (x, y, z, w)
def rotvec_to_quat(rotvec):
    """
    Convert a rotation vector (axis-angle / exponential coordinates)
    to quaternion [x, y, z, w]. 
    Handles both numpy arrays and torch tensors as input.
    """
    try:
        import torch
        if isinstance(rotvec, torch.Tensor):
            rotvec = rotvec.detach().cpu().numpy()
    except ImportError:
        pass
    rotvec = np.array(rotvec, copy=True)  # ensure writeable
    r = R.from_rotvec(rotvec)
    return r.as_quat()   # returns [x, y, z, w]


# Apply rotation vector to quaternion (x, y, z, w)
def apply_rotvec_to_quat(current_quat, rotvec):
    """
    Apply a rotation given as rotvec (axis-angle vector) to current_quat. 
    The function assumes the rotation part of Libero is given as an 
    axis-angle vector (i.e., a rotation vector r = θ * u), which is the Libero convention. 
    The magnitude ||r|| = θ is the rotation angle in radians.
    
    Args:
        current_quat: (4,) quaternion [x,y,z,w]
        rotvec: (3,) rotation vector (theta * axis)
    
    Returns:
        new_quat: (4,) quaternion [x,y,z,w]
    """
    r_current = R.from_quat(current_quat)
    r_delta = R.from_rotvec(rotvec)
    # Compose: new = delta * current
    r_new = r_delta * r_current
    return r_new.as_quat()


def prepare_libero_observation(ee_pos, ee_quat_wxyz, gripper_value, use_rotvec=True):
    """
    Prepare observation for Libero policy from Franka EE pose and gripper.
    
    Args:
        ee_pos: array-like (3,), end-effector position [x, y, z]
        ee_quat_wxyz: array-like (4,), quaternion [w, x, y, z]
        gripper_value: float, current gripper state (0-1 or -1 to 1)
        use_rotvec: bool, if True convert quaternion to rotation vector (axis-angle),
                    else keep quaternion in observation.
                    
    Returns:
        obs: np.ndarray, observation vector for Libero policy
             [x, y, z, dRx, dRy, dRz, g] if use_rotvec
             or [x, y, z, qw, qx, qy, qz, g] if not
    """
    ee_pos = np.asarray(ee_pos, dtype=float)
    gripper_value = float(gripper_value)
    if use_rotvec:
        # Convert quaternion to rotation vector (axis-angle)
        # Reorder from [w, x, y, z] to [x, y, z, w] for SciPy
        quat_xyzw = np.array([ee_quat_wxyz[1], ee_quat_wxyz[2],
                              ee_quat_wxyz[3], ee_quat_wxyz[0]])
        r = R.from_quat(quat_xyzw)
        rotvec = r.as_rotvec()  # 3D rotation vector
        obs = np.concatenate([ee_pos, rotvec, gripper_value, gripper_value])
    else:
        # Keep quaternion as is (w,x,y,z)
        obs = np.concatenate([ee_pos, ee_quat_wxyz, [gripper_value]])
    return obs


def libero_action_to_pose(current_pos, current_quat, action):
    """
    Convert Libero action into a new EE pose (position, quaternion).
    
    Args:
        current_pos: (3,) current EE position
        current_quat: (4,) current EE quaternion [x,y,z,w]
        action: (7,) [dx, dy, dz, dRx, dRy, dRz, g]
    
    Returns:
        new_pos: (3,) numpy array
        new_quat: (4,) numpy array [x,y,z,w]
        gripper_cmd: scalar
    """
    if action.shape[0] != 7:
        raise ValueError("libero_action must be length 7: [dx,dy,dz,dRx,dRy,dRz,g]")
    # Translation update
    delta_pos = action[:3]
    new_pos = current_pos + delta_pos

    # Rotation update
    rotvec = action[3:6]
    new_quat = rotvec_to_quat(rotvec)

    gripper_cmd = action[6]
    return new_pos, new_quat, gripper_cmd


def delta_to_joint_velocities(
    ee_entity,
    end_effector,
    q_current,
    action,
    integration_dt=0.1,
    Kpos=0.95,
    Kori=0.95,
    damping=0.05,
    Kn=None,
):
    """
    Convert delta end-effector command [dx, dy, dz, dRx, dRy, dRz]
    into joint velocities using damped least squares IK.
    Uses Genesis RigidEntity.get_jacobian() from the *end-effector entity*.

    Args:
        ee_entity (RigidEntity): Genesis end-effector entity.
        q_current (np.ndarray): current joint positions of the robot (N,).
        dx, dy, dz (float): Cartesian deltas.
        dRx, dRy, dRz (float): small rotation deltas (axis-angle, radians).
        integration_dt (float): timestep.
        Kpos (float): position gain.
        Kori (float): orientation gain.
        damping (float): damping coefficient λ for pseudo-inverse.
        Kn (np.ndarray or None): nullspace gains (N,). Defaults to ones.

    Returns:
        np.ndarray: joint velocities dq (N,)
    """
    # Get Jacobian of EEF (6 x N)
    J = ee_entity.get_jacobian(end_effector)
    dx, dy, dz, dRx, dRy, dRz = action[:6]

    n_joints = J.shape[1]
    if Kn is None:
        Kn = np.ones(n_joints)

    # Build 6D twist (scaled velocity target)
    twist = np.zeros(6)
    # twist[:3] = Kpos * np.array([dx, dy, dz]) / integration_dt
    # twist[3:] = Kori * np.array([dRx, dRy, dRz]) / integration_dt
    twist[:3] = Kpos * np.array([dx, dy, dz])
    twist[3:] = Kori * np.array([dRx, dRy, dRz])

    n_arm_joints = 7
    J_np = J.detach().cpu().numpy()[:, :n_arm_joints]
    JJt = J_np @ J_np.T
    lambda_I = (damping ** 2) * np.eye(6)
    dq = J_np.T @ np.linalg.solve(JJt + lambda_I, twist)

    # Add nullspace posture control (only 7×7)
    I = np.eye(n_arm_joints)
    dq += (I - np.linalg.pinv(J_np) @ J_np) @ (Kn[:n_arm_joints] * q_current[:n_arm_joints])
    return dq


def pos_rnd(sign1, sign2):
    return  random.choice([sign1, sign2])*random.uniform(0.17, 0.42)


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def rpy_to_quat(roll, pitch, yaw):
    """
    Converts roll, pitch, yaw (in radians) to quaternion (w, x, y, z).
    
    Args:
        roll, pitch, yaw : float
            Rotation angles in radians.
    Returns:
        np.array([w, x, y, z])
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quat_rotate(quat, vec):
    """Rotate vector `vec` by quaternion `quat = [w, x, y, z]`."""
    w, x, y, z = quat
    q = np.array([w, x, y, z])
    v = np.array([0, *vec])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    qv = np.array([
        q[0]*v[0] - q[1]*v[1] - q[2]*v[2] - q[3]*v[3],
        q[0]*v[1] + q[1]*v[0] + q[2]*v[3] - q[3]*v[2],
        q[0]*v[2] - q[1]*v[3] + q[2]*v[0] + q[3]*v[1],
        q[0]*v[3] + q[1]*v[2] - q[2]*v[1] + q[3]*v[0]
    ])
    result = np.array([
        qv[0]*q_conj[0] - qv[1]*q_conj[1] - qv[2]*q_conj[2] - qv[3]*q_conj[3],
        qv[0]*q_conj[1] + qv[1]*q_conj[0] + qv[2]*q_conj[3] - qv[3]*q_conj[2],
        qv[0]*q_conj[2] - qv[1]*q_conj[3] + qv[2]*q_conj[0] + qv[3]*q_conj[1],
        qv[0]*q_conj[3] + qv[1]*q_conj[2] - qv[2]*q_conj[1] + qv[3]*q_conj[0]
    ])
    return result[1:4]