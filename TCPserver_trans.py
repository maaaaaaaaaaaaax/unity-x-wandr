# ============================================================================
# TCP Server for Real-time Motion Generation and Transmission
# ============================================================================
# This script implements a TCP server that receives body pose data from Unity,
# processes it through a motion generation model (WANDR), and sends back the
# predicted motion for real-time animation in Unity.
# ============================================================================

import socket          # TCP/IP networking
import torch           # PyTorch for model inference
import numpy as np     # Numerical operations
import time            # Time delays for frame-rate control
import json            # JSON serialization for network messages
import os              # File system operations
import sys             # System utilities
import traceback       # Exception traceback printing

from scipy.spatial.transform import Rotation as R  # Rotation conversions
import smplx           # SMPL-X body model
import pyrender        # 3D rendering library
import trimesh         # 3D mesh processing
from utils.misc import cast_dict_to_numpy, cast_dict_to_tensors  # Type casting utilities
from rendering.render_utils import render_motion  # Video rendering
from aitviewer.headless import HeadlessRenderer  # Headless rendering
from aitviewer.configuration import CONFIG as C  # AITViewer configuration
from models.base_motion_prior import Human       # WANDR motion model
from utils.transformations import transform_body_pose  # Pose representation conversions

# Set global random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.cuda.is_available() and torch.cuda.manual_seed_all(0)


# ============================================================================
# Joint Definitions
# ============================================================================
# Mapping of joint names to indices in SMPL-X and Unity body models
# Both models use the same joint structure (21 joints for body, excluding hands)

# SMPL-X joint name to index mapping for 22-joint body skeleton
smpljoints={'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3, 'left_knee': 4, 'right_knee': 5, 
            'spine2': 6, 'left_ankle': 7, 'right_ankle': 8, 'spine3': 9, 'left_foot': 10, 
            'right_foot': 11, 'neck': 12, 'left_collar': 13, 'right_collar': 14, 'head': 15, 
            'left_shoulder': 16, 'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19, 
            'left_wrist': 20, 'right_wrist': 21}  # Mapping for SMPL-X body model

# Unity SMPL-X joint name to index mapping (same structure as SMPL-X)
unityjoints={ "pelvis":0,"left_hip":1,"right_hip":2,"spine1":3,"left_knee":4,"right_knee":5,
            "spine2":6,"left_ankle":7,"right_ankle":8,"spine3":9, "left_foot":10,
            "right_foot":11,"neck":12,"left_collar":13,"right_collar":14,"head":15,
            "left_shoulder":16,"right_shoulder":17,"left_elbow":18, "right_elbow":19,
            "left_wrist":20,"right_wrist":21}  # Mapping for Unity body model

# ============================================================================
# Model Configuration and Loading
# ============================================================================

# Path to SMPL-X pre-trained body model files
smplxmodel= './data/body_models'

# Gender for SMPL-X body model (affects mesh shape)
gender = 'female'

# Compute device: CPU for compatibility, GPU (cuda) for faster inference
DEVICE = torch.device("cpu")

# Path to pre-trained WANDR motion generation model checkpoint
MODEL_FILE = './model_weights/wandr.ckpt'

# Output directory for saving generated motion sequences
MOTION_OUTPUT_DIR = './mouse_click_output'

# Flag to use live pose from Unity or fallback to fixed initialization pose
# Use the pose currently shown in Unity as the initial state for every WANDR
# rollout. Set this to False to fall back to deps/init_pose.npz while comparing
# the corrected live-pose conversion against the previous behaviour.
USE_LIVE_POSE_INITIALIZATION = False

# Verify model file exists before attempting to load
if os.path.exists(MODEL_FILE):
    print('Model loaded successfully')
else:
    print(f"Could not find model at {MODEL_FILE}")
    sys.exit(1)

# Load and initialize WANDR motion prior model for inference
# The motion model learns to predict natural human motion from current state and goal
model: Human = Human.load_from_checkpoint(MODEL_FILE, weights_only=False)
model.eval()  # Set to evaluation mode (disables dropout, batch normalization, etc.)



# ============================================================================
# Coordinate System Transformation Matrices
# ============================================================================
# Unity uses a left-handed coordinate system (x right, y up, z forward)
# WANDR/SMPL uses a right-handed coordinate system (z up)
# This matrix transforms from Unity coordinates to WANDR coordinates:
#   x_w = -x_u  (flip X)
#   y_w = -z_u  (Unity forward becomes negative WANDR Y)
#   z_w =  y_u  (Unity height becomes WANDR Z)

UNITY_TO_WANDR = torch.tensor(
    [[-1.0, 0.0, 0.0],  # Row 1: maps Unity X to WANDR X (negated)
     [ 0.0, 0.0, -1.0], # Row 2: maps Unity Z to WANDR Y (negated)
     [ 0.0, 1.0, 0.0]], # Row 3: maps Unity Y to WANDR Z
    dtype=torch.float32
)

# Unity world coordinates and the local coordinate system of the imported
# SMPL-X skeleton are not the same basis conversion:
#
#   Unity world -> WANDR world:
#       handled by UNITY_TO_WANDR above (Y-up/left-handed -> Z-up/right-handed)
#
#   Unity SMPL-X local -> canonical SMPL-X local:
#       only the X axis changes sign. This convention is documented by
#       SMPLX.QuatFromRodrigues() in the bundled Unity SMPL-X package.
#
# A SMPL-X global orientation maps the canonical body basis into the world
# basis. Consequently its conversion needs the world matrix on the left and
# this model-basis matrix on the right. Local joint rotations use this matrix
# on both sides.
UNITY_MODEL_TO_SMPLX = torch.tensor(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 1.0, 0.0],
     [ 0.0, 0.0, 1.0]],
    dtype=torch.float32
)


# ============================================================================
# SMPL-X Pose Expansion Functions
# ============================================================================

def _deprecated_map_21_to_55(pose_21):
    """
    Expand 21-joint body pose to 55-joint SMPL-X pose.
    
    SMPL-X has 55 joints:
    - Joint 0: Pelvis (root)
    - Joints 1-21: Body joints (already in pose_21)
    - Joints 22: Jaw
    - Joints 23-24: Eyes
    - Joints 25-39: Left hand (15 joints)
    - Joints 40-54: Right hand (15 joints)
    
    Args:
        pose_21: numpy array of shape (21, 6) containing 6D rotation representations
        
    Returns:
        pose_55: numpy array of shape (55, 6) with expanded hand and eye joints
    """
    pose_55 = np.zeros((55, 6))  # Initialize empty array for 55 joints
    pose_21 = pose_21.reshape(21, 6)  # Ensure pose_21 is reshaped to 21x6
    pose_55[1:22] = pose_21  # Copy the first 21 joints directly (body)
    
    # Pelvis (joint 0) — estimate as midpoint between hips
    pose_55[0] = (pose_55[1] + pose_55[2]) / 2  # Average of left_hip and right_hip

    # Jaw (joint 22) — estimate as midpoint between shoulders
    pose_55[22] = (pose_55[16] + pose_55[17]) / 2  # Jaw (estimated between shoulders)

    # Eyes (joints 23-24) — estimate positions around head with slight offset
    pose_55[23] = pose_55[15] + np.array([0.02, 0, 0, 0, 0, 0])  # left_eye_smplhf
    pose_55[24] = pose_55[15] + np.array([-0.02, 0, 0, 0, 0, 0])  # right_eye_smplhf

    # Hand joints — estimate finger positions based on wrist position
    pose_55[25:40] = interpolate_fingers(pose_55[20])  # left hand (15 joints)
    pose_55[40:55] = interpolate_fingers(pose_55[21])  # right hand (15 joints)
    
    return pose_55

def interpolate_fingers(wrist_data):
    """
    Generate estimated finger joint positions based on wrist position and rotation.
    
    SMPL-X hands have 15 joints per hand (5 fingers × 3 joints per finger).
    Since finger data is not provided, we estimate positions linearly from wrist.
    
    Args:
        wrist_data: numpy array of shape (6,) containing wrist position (3) and rotation (3)
        
    Returns:
        finger_joints: numpy array of shape (15, 6) with estimated finger joint rotations
    """
    finger_joints = np.zeros((15, 6))  # 5 fingers * 3 joints each = 15
    wrist_pos, wrist_rot = wrist_data[:3], wrist_data[3:]  # Extract position and rotation

    for i in range(15):
        # Linear interpolation weight increases for distal finger joints
        weight = (i % 3) / 2.0  # 3 joints per finger, ranges from 0 to 1.5
        # Estimate finger joint positions extending from wrist in X direction
        finger_joints[i, :3] = wrist_pos + weight * np.array([0.1, 0, 0])  # Example X translation
        # Use wrist rotation for all finger joints (simplified)
        finger_joints[i, 3:] = wrist_rot  # Same rotation for simplicity

    return finger_joints


def generate_smplx_model(data):
    """
    Generate and visualize a 3D SMPL-X model from pose data.
    
    This function creates a 3D mesh from motion data and displays it using
    PyRender's interactive viewer.
    
    Args:
        data: Dictionary containing body pose parameters (body_pose, body_orient, body_transl)
    """
    # Load SMPL-X body model with 55 joints (including hands and face)
    smplx_model = smplx.create(smplxmodel, model_type="smplx", gender=gender, use_pca=False, batch_size=1)
    print('smpl model loaded')
    
    # Extract and convert body pose to tensor (skip pelvis joint 0, use joints 1-21)
    body_pose = torch.tensor(np.array(data['body_pose'][1:22]), dtype=torch.float32).unsqueeze(0)  # (1, 21*3)
    
    # Extract and convert global orientation (root rotation) to tensor
    global_orient = torch.tensor(np.array(data['body_orient']), dtype=torch.float32).unsqueeze(0)   # Root rotation (1, 3)
    
    # Extract and convert translation (body position) to tensor
    transl = torch.tensor(np.array(data['body_transl']), dtype=torch.float32).unsqueeze(0)          # Position (1, 3)

    # Generate 3D mesh vertices using SMPL-X model forward pass
    output = smplx_model(global_orient=global_orient, body_pose=body_pose, transl=transl)
    vertices = output.vertices.detach().cpu().numpy().squeeze()  # Extract vertices from GPU
    faces = smplx_model.faces  # Get mesh face indices

    # Create mesh object from vertices and faces
    mesh = trimesh.Trimesh(vertices, faces)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    
    # Create scene and add mesh
    scene = pyrender.Scene()
    scene.add(mesh)
    
    # Launch interactive 3D viewer with lighting and world axes
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=True)

    return


# ============================================================================
# Coordinate System Conversion Functions
# ============================================================================

def convert_root_orientation_unity_to_wandr(pose6d: np.ndarray) -> np.ndarray:
    """
    Convert a Unity GameObject/root orientation to WANDR global orientation.

    A root/global orientation maps the canonical model basis into the world
    basis, so its two sides require different basis transforms:

        R_wandr = A * R_unity * C

    where A converts Unity world coordinates to WANDR world coordinates and C
    converts Unity's imported SMPL-X model basis to canonical SMPL-X.
    
    Args:
        pose6d: numpy array of shape (J, 6) or (6,) containing 6D rotation representations
        
    Returns:
        Converted 6D rotations in WANDR coordinate system
    """
    # Reshape input to batch format if needed
    p = torch.tensor(pose6d, dtype=torch.float32).reshape(-1, 6)
    
    # Convert 6D representation to full 3x3 rotation matrix
    Ru = transform_body_pose(p, "6d->rot")
    
    # Get coordinate transformation matrices on the device where data lives
    A = UNITY_TO_WANDR.to(Ru.device)
    C = UNITY_MODEL_TO_SMPLX.to(Ru.device)
    
    # Apply basis transformation: world transform on left, model basis on right
    Rw = A @ Ru @ C
    
    # Convert result back to 6D representation for transmission
    pw = transform_body_pose(Rw, "rot->6d")
    
    return pw.detach().cpu().numpy()


def convert_local_pose6d_unity_to_wandr(pose6d: np.ndarray) -> np.ndarray:
    """
    Convert Unity SMPL-X local joint rotations to canonical SMPL-X/WANDR coordinates.
    
    Local joint rotations (as opposed to global/root orientation) only need the
    model-basis transformation on both sides.
    
    Args:
        pose6d: numpy array of local joint rotations in 6D representation
        
    Returns:
        Joint rotations converted to WANDR coordinate system
    """
    # Reshape to batch format
    p = torch.tensor(pose6d, dtype=torch.float32).reshape(-1, 6)
    
    # Convert 6D to full rotation matrix
    Ru = transform_body_pose(p, "6d->rot")
    
    # Apply model basis transformation on both sides (local joint frame)
    C = UNITY_MODEL_TO_SMPLX.to(Ru.device)
    Rw = C @ Ru @ C.T  # Conjugate transpose for local transformations
    
    # Convert back to 6D representation
    pw = transform_body_pose(Rw, "rot->6d")
    
    return pw.detach().cpu().numpy()

def to_float_array(s: str):
    """
    Parse comma-separated string into numpy float32 array.
    
    Args:
        s: String containing comma-separated float values
        
    Returns:
        numpy array of float32 values
    """
    return np.fromstring(s.strip(), sep=",", dtype=np.float32)

def convert_translation_unity_to_wandr(t_xyz):
    """
    Convert 3D translation from Unity coordinates to WANDR coordinates.
    
    Args:
        t_xyz: numpy array of shape (3,) or (1, 3) containing translation vector
        
    Returns:
        Translated vector in WANDR coordinate system
    """
    t = torch.tensor(t_xyz, dtype=torch.float32).reshape(1, 3)
    # Apply transformation: t_w = UNITY_TO_WANDR @ t_u
    t_w = (UNITY_TO_WANDR @ t.T).T
    return t_w.squeeze(0).numpy()


def convert_translation_wandr_to_unity(t_xyz):
    """
    Convert 3D translation from WANDR coordinates to Unity coordinates (inverse transform).
    
    Since UNITY_TO_WANDR is an orthogonal transformation matrix, its inverse
    is simply its transpose.
    
    Args:
        t_xyz: numpy array containing translation in WANDR coordinates
        
    Returns:
        Translation vector in Unity coordinate system
    """
    t = torch.tensor(t_xyz, dtype=torch.float32).reshape(-1, 3)
    
    # Apply inverse transformation: t_u = UNITY_TO_WANDR^T @ t_w
    WANDR_TO_UNITY = UNITY_TO_WANDR.T
    t_u = (WANDR_TO_UNITY @ t.T).T

    return t_u.numpy()

def convert_root_orientation_wandr_to_unity(pose6d: np.ndarray) -> np.ndarray:
    """
    Convert WANDR global orientation to a Unity GameObject/root orientation (inverse).

    Inverse transformation formula:
        R_unity = A^T * R_wandr * C^T
    
    Args:
        pose6d: numpy array containing 6D rotation representations in WANDR coordinates
        
    Returns:
        Converted 6D rotations in Unity coordinate system
    """
    # Reshape to batch format
    p = torch.tensor(pose6d, dtype=torch.float32).reshape(-1, 6)
    
    # Convert 6D to full rotation matrix
    Rw = transform_body_pose(p, "6d->rot")
    
    # Apply the inverse world/model basis conversion
    A = UNITY_TO_WANDR.to(Rw.device)
    C = UNITY_MODEL_TO_SMPLX.to(Rw.device)
    
    # Inverse: transpose the transformation matrices
    Ru = A.T @ Rw @ C.T
    
    # Convert back to 6D representation
    pu = transform_body_pose(Ru, "rot->6d")
    
    return pu.detach().cpu().numpy()

def quat_xyzw_to_rot6d_np(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to 6D rotation representation.
    
    Conversion path: Quaternion -> Axis-angle -> 6D representation
    
    Args:
        q_xyzw: numpy array of shape (N, 4) or (4,) containing quaternions in (x,y,z,w) format
        
    Returns:
        6D rotation representation
    """
    q = np.asarray(q_xyzw, dtype=np.float32).reshape(-1, 4)
    # Convert quaternion to axis-angle representation
    aa = R.from_quat(q).as_rotvec().astype(np.float32)  # (N, 3)
    aa_t = torch.from_numpy(aa)
    # Convert axis-angle to 6D representation
    r6 = transform_body_pose(aa_t, "aa->6d")
    return r6.reshape(-1, 6).cpu().numpy()

def rot6d_to_quat_xyzw_np(r6: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to quaternion (x, y, z, w).
    
    Conversion path: 6D representation -> Axis-angle -> Quaternion
    
    Args:
        r6: numpy array of shape (N, 6) or (6,) containing 6D rotations
        
    Returns:
        Quaternion in (x, y, z, w) format
    """
    r6_t = torch.tensor(r6, dtype=torch.float32).reshape(-1, 6)
    # Convert 6D to axis-angle representation
    aa = transform_body_pose(r6_t, "6d->aa")  # (...,3)
    # Convert axis-angle to quaternion
    quat = R.from_rotvec(aa.detach().cpu().numpy()).as_quat().astype(np.float32)  # (x,y,z,w)
    return quat


def smplx_rot6d_to_unity_quat_xyzw_np(r6: np.ndarray) -> np.ndarray:
    """
    Convert local SMPL-X joint rotations to Unity SMPL-X package quaternions.

    The Unity SMPL-X package maps Rodrigues/axis-angle with:
        axis = (-rodX, rodY, rodZ)
        angle = -|rod|
    That is equivalent to converting SMPL-X axis-angle to a quaternion and
    flipping the y and z vector components: (x, -y, -z, w).

    This is intentionally different from convert_root_orientation_wandr_to_unity(), which
    is a global coordinate-frame transform for root orientation. Local body_pose
    joints live in the SMPL-X joint basis and need the SMPL-X Unity convention.
    """
    quat = rot6d_to_quat_xyzw_np(r6).reshape(-1, 4)
    quat[:, 1] *= -1.0
    quat[:, 2] *= -1.0
    return quat.astype(np.float32)



def _deprecated_extract_orient6d_and_bodypose6d(pose_flat: np.ndarray, ):
    """
    Extract root orientation and body pose from flattened pose array.
    
    Supports multiple pose formats:
    - 220: 55 joints × 4 (quaternion)
    - 132: 22 joints × 6 (6D representation)
    - 126: 21 joints × 6 (body only, no root)
    - 330: 55 joints × 6 (6D representation)
    
    Args:
        pose_flat: Flattened pose array
        
    Returns:
        Tuple of (orient6d_w, body21_6d_w): root orientation and body pose in WANDR coords
    """
    raise RuntimeError(
        "Deprecated parser: Unity Pose must not provide root orientation. "
        "Use extract_bodypose6d_from_unity_pose() and the explicit Orientation field instead."
    )

    n = pose_flat.size

    # Format: 55 joints × 4 quaternions = 220 values
    if n == 220:
        q55 = pose_flat.reshape(55, 4)
        # Extract root (pelvis) quaternion and convert to 6D
        orient6d = quat_xyzw_to_rot6d_np(q55[0:1])[0]
        # Extract body joints (1-21) and convert to 6D
        body21_6d = quat_xyzw_to_rot6d_np(q55[1:22])

    # Format: 22 joints × 6 (6D representation)
    elif n == 132:
        p22 = pose_flat.reshape(22, 6)
        # First joint is root orientation
        orient6d = p22[0]
        # Remaining 21 joints are body pose
        body21_6d = p22[1:22]

    # Format: 21 joints × 6 (body only, no root orientation)
    elif n == 126:
        body21_6d = pose_flat.reshape(21, 6)
        # Do not invent a root orientation; use identity (no rotation)
        orient6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # Format: 55 joints × 6 (6D representation)
    elif n == 330:
        p55 = pose_flat.reshape(55, 6)
        # Extract root (pelvis) orientation
        orient6d = p55[0]
        # Extract body joints (1-21)
        body21_6d = p55[1:22]
    else:
        raise ValueError(f"Unsupported Pose length: {n}. Expected 126, 132, 220, or 330 values.")

    # Convert both to WANDR coordinate system
    orient6d_w = convert_root_orientation_unity_to_wandr(orient6d.reshape(1, 6)).reshape(6)
    body21_6d_w = convert_local_pose6d_unity_to_wandr(body21_6d).reshape(-1)
    return orient6d_w, body21_6d_w


def extract_bodypose6d_from_unity_pose(pose_flat: np.ndarray) -> np.ndarray:
    """
    Extract only WANDR's SMPL-X body_pose input from Unity's Pose field.

    In this project the TCP message already has a separate Orientation field for
    the Unity GameObject/root rotation. Therefore Pose must not be treated as
    another source for body_orient. Pose is only used for local body joints.

    Supported Unity payloads:
    - 220: 55 joints x quaternion (x,y,z,w). Joint 0 is pelvis/local root and is
      skipped; joints 1..21 become WANDR body_pose.
    - 126: 21 joints x 6D body pose. Used directly as body_pose.

    The 132/330 formats are intentionally not accepted here because they are
    ambiguous in this Unity pipeline: the first 6D entry could be interpreted as
    a root orientation, but root orientation is already sent separately.
    """
    n = pose_flat.size

    if n == 220:
        q55 = pose_flat.reshape(55, 4)
        body21_6d = quat_xyzw_to_rot6d_np(q55[1:22])
    elif n == 126:
        body21_6d = pose_flat.reshape(21, 6)
    else:
        raise ValueError(
            f"Unsupported Unity Pose length: {n}. "
            "Expected 220 values (55 quaternions) or 126 values (21 body-joint 6D rotations)."
        )

    return convert_local_pose6d_unity_to_wandr(body21_6d).reshape(-1)

def parse_unity_message(message, ):
    """
    Parse pose and click data from Unity TCP message.
    
    Message format (semicolon-separated):
    "MouseClick:x,y,z;Translation:x,y,z;Orientation:qx,qy,qz,qw;Pose:r6d_values..."
    
    Args:
        message: String message from Unity containing pose data
        
    Returns:
        Dictionary with keys: mouse_click, body_transl, body_orient, body_pose
        (All converted to WANDR coordinates)
    """
    data = {"mouse_click": None, "body_transl": None, "body_orient": None, "body_pose": None}
    try:
        parts = message.split(";")
        if len(parts) < 4:
            return data

        # Parse mouse click position
        if parts[0].startswith("MouseClick:"):
            mouse_click = to_float_array(parts[0].split(":", 1)[1])
            if mouse_click.size >= 3:
                # Convert from Unity to WANDR coordinates
                data["mouse_click"] = convert_translation_unity_to_wandr(mouse_click[:3]).tolist()

        # Parse body translation (position)
        if parts[1].startswith("Translation:"):
            transl = to_float_array(parts[1].split(":", 1)[1])
            if transl.size >= 3:
                # Convert from Unity to WANDR coordinates
                data["body_transl"] = convert_translation_unity_to_wandr(transl[:3]).tolist()

        # Parse body orientation (root rotation)
        if parts[2].startswith("Orientation:"):
            orient = to_float_array(parts[2].split(":", 1)[1])
            # Unity sends quaternion (x,y,z,w) format
            if orient.size == 4:
                # Convert quaternion to 6D, then to WANDR coordinates
                o6 = quat_xyzw_to_rot6d_np(orient.reshape(1, 4)).reshape(1, 6)
                data["body_orient"] = convert_root_orientation_unity_to_wandr(o6).reshape(6).tolist()

        # Parse full body pose
        if parts[3].startswith("Pose:"):
            pose_info = to_float_array(parts[3].split(":", 1)[1])
            # Orientation comes from the explicit Orientation field above.
            # Pose contributes only WANDR body_pose joints 1..21.
            bodypose_flat_w = extract_bodypose6d_from_unity_pose(pose_info)
            data["body_pose"] = bodypose_flat_w.tolist()


    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())
    return data


# This definition intentionally overrides the older map_21_to_55 helper above.
# Keep it close to send_motion_to_unity(), because this is the version used when
# frames are converted back to Unity.
def map_21_to_55(pose_21):
    """
    Expand WANDR's 21 generated body joints to the 55 SMPL-X joints expected by Unity.

    Unsupported SMPL-X joints stay at identity. The original helper tried to infer
    jaw, eye, and finger rotations by adding offsets to 6D rotation values; those
    values are rotations, not positions, so the result can become invalid and jittery.
    """
    pose_21 = pose_21.reshape(21, 6)

    # 6D identity rotation: first two columns of a 3x3 identity rotation matrix.
    # This is the safe rest value for SMPL-X joints that WANDR did not generate.
    identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # SMPL-X expects 55 joints. WANDR only gives the 21 body joints below the root,
    # so start every joint at rest and overwrite only the supported body range.
    pose_55 = np.tile(identity_6d, (55, 1)).astype(np.float32)

    # Unity/SMPL-X joint 0 is the pelvis/root. The generated body pose maps to
    # joints 1..21; root orientation is sent separately as body_orient.
    pose_55[1:22] = pose_21
    return pose_55


def send_motion_to_unity(conn, motion_data ):
    """
    Send generated motion sequence to Unity at 30 FPS.
    
    Converts WANDR motion to Unity format and sends frame-by-frame.
    Each frame includes position, root rotation, and all body pose rotations.
    
    Args:
        conn: TCP socket connection to Unity
        motion_data: Dictionary containing motion sequence (body_transl, body_orient, body_pose)
    """
    try:
        # Extract motion data and ensure correct shapes
        body_transl = motion_data["body_transl"].squeeze(1)   # (S,3) - sequence of positions
        body_orient = motion_data["body_orient"].squeeze(1)   # (S,6) - root rotations
        body_pose = motion_data["body_pose"].squeeze(1)       # (S,126) - body joint rotations

        n_frames = body_transl.shape[0]
        
        # Convert from WANDR to Unity coordinates. Translation, root orientation,
        # and local body joints are converted separately so Unity can apply the
        # root transform on the GameObject and the body pose on the skeleton.
        transl_unity = convert_translation_wandr_to_unity(body_transl)
        orient_unity_6d = convert_root_orientation_wandr_to_unity(body_orient)

        # Send each frame to Unity at 30 FPS (1/30 = 0.033s delay between frames)
        for i in range(n_frames):
            # Expand 21 generated body joints to the full SMPL-X skeleton.
            # Unsupported face/hand joints stay at identity to avoid artificial
            # rotation noise.
            pose55_6d = map_21_to_55(body_pose[i].reshape(-1))

            # IMPORTANT: Do not apply root rotation twice. Unity receives root
            # orientation as body_orient, so pelvis/local joint 0 must stay identity.
            pose55_6d[0] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)  # Identity rotation

            # Convert local SMPL-X joint rotations to the quaternion convention
            # used by the Unity SMPL-X package.
            q55 = smplx_rot6d_to_unity_quat_xyzw_np(pose55_6d).reshape(-1)

            # Root orientation is sent separately so SMPLXController can choose to
            # use it, ignore it, or smooth it independently from the joint pose.
            q_root = rot6d_to_quat_xyzw_np(orient_unity_6d[i]).reshape(-1)

            # Create JSON frame data
            frame_data = {
                "body_transl": transl_unity[i].tolist(),    # Position (3,)
                "body_orient": q_root.tolist(),             # Root rotation (4,)
                "body_pose": q55.tolist()                   # Full body pose (220,)
            }
            
            # Newline is the frame delimiter used by TCPManager.cs. TCP itself is
            # stream-based, so Unity reconstructs complete frames by splitting on
            # this newline.
            conn.sendall((json.dumps(frame_data) + "\n").encode("utf-8"))
            # Sleep to maintain 30 FPS transmission rate
            time.sleep(1.0 / 30.0)

    except Exception as e:
        print(f"[ERROR] Failed to send motion to Unity: {e}")
        print(traceback.format_exc())




def receive_unity_messages(conn):
    """
    Yield complete newline-delimited messages from Unity socket.

    TCP is stream-based, so a single recv() can contain a partial message,
    multiple complete messages, or a mix. Unity terminates each message with '\n',
    so we buffer incomplete chunks and yield complete lines.
    
    Args:
        conn: TCP socket connection
        
    Yields:
        Strings containing complete parsed messages from Unity
    """
    buffer = ""  # Accumulate incoming bytes
    
    while True:
        # Receive up to 8KB from socket
        chunk = conn.recv(8192)
        
        # Empty chunk means connection closed
        if not chunk:
            # Yield any remaining buffered data
            if buffer.strip():
                yield buffer.strip()
            return

        # Decode bytes to string and append to buffer
        buffer += chunk.decode("utf-8")
        
        # Extract and yield all complete messages (lines ending with '\n')
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)  # Split on first newline
            line = line.strip()
            if line:  # Skip empty lines
                yield line


def start_server():
    """
    Main TCP server loop for real-time motion generation.
    
    This function:
    1. Starts a TCP server listening on localhost:5005
    2. Waits for and accepts connection from Unity client
    3. Continuously receives pose and click data from Unity
    4. Processes data through WANDR motion generation model
    5. Generates 8-second motion sequences (240 frames at 30 FPS)
    6. Sends predicted motion back to Unity frame-by-frame
    7. Saves generated motions to files for analysis/debugging
    
    The server runs continuously until interrupted (Ctrl+C) or connection is lost.
    All coordinate transformations between Unity and WANDR are handled internally.
    """
    # Server configuration
    HOST, PORT = "127.0.0.1", 5005
    
    # Create TCP socket and bind to port
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)  # Allow 1 pending connection

    print(f"Waiting for Unity on {HOST}:{PORT}...")
    conn, addr = server.accept()
    print(f"Connected to {addr}")
    
    ix = 0  # Counter for saved motion file numbering
    
    try:
        # Main loop: receive messages from Unity
        for data in receive_unity_messages(conn):
            # Parse received message into pose components and convert to WANDR coords
            parsed_data = parse_unity_message(data)

            # Check if all required pose components were received successfully
            if (parsed_data["body_transl"] is not None and
                parsed_data["body_orient"] is not None and
                parsed_data["body_pose"] is not None):

                # Extract goal position (where user clicked in Unity)
                goal_np = parsed_data["mouse_click"]
                # Force height (z in WANDR coords) to always be 1 meter (walking on ground)
                goal_np = [goal_np[0], goal_np[1], 1]

                # Convert goal to tensor on model device
                goal = torch.tensor(goal_np, dtype=torch.float32, device=model.device).reshape(1, 3)
                print(f"[DEBUG] Received goal from Unity (WANDR coords): {goal_np} ")
                print(f"[DEBUG] Wandr goal tensor: {goal} - Shape: {goal.shape}\n")

                # Convert current pose components to tensors on model device
                body_transl_tensor = torch.tensor(parsed_data["body_transl"], dtype=torch.float32, device=model.device).reshape(1, 1, 3)
                body_orient_tensor_6d = torch.tensor(parsed_data["body_orient"], dtype=torch.float32, device=model.device).reshape(1, 1, 6)
                body_pose_tensor_6d = torch.tensor(parsed_data["body_pose"], dtype=torch.float32, device=model.device).reshape(1, 1, 21 * 6)

                # Convert rotations from 6D representation to axis-angle (WANDR's expected format)
                body_orient_tensor_aa = transform_body_pose(body_orient_tensor_6d, "6d->aa")    
                body_pose_tensor_aa = transform_body_pose(body_pose_tensor_6d, "6d->aa")

                # Prepare initial state for motion generation
                if USE_LIVE_POSE_INITIALIZATION:
                    # Use the current pose from Unity as starting point
                    # Keep all components from the same frame together for physical coherence
                    init_state_dict = {
                        "body_transl": body_transl_tensor,
                        "body_orient": body_orient_tensor_aa,
                        "body_pose": body_pose_tensor_aa
                    }
                else:
                    # Fallback: use pre-recorded initial pose, only update position
                    # Useful for comparing coordinate conversion changes
                    init_state = np.load('./deps/init_pose.npz')
                    init_state = {key: init_state[key] for key in init_state.files}
                    init_state_dict = cast_dict_to_tensors(init_state, device=model.device)
                    init_state_dict['body_transl'] = body_transl_tensor
                    init_state_dict['body_orient'] = body_orient_tensor_aa

                # Generate motion sequence: 8 seconds at 30 FPS = 240 frames
                motion_duration = 8
                out = model.rollout(
                    init_state_dict,
                    goal,
                    n_steps=motion_duration * 30,  # 240 frames
                    return_smpl_joints=True,
                    angle_format='aa'  # Use axis-angle representation
                )
                
                # Convert output tensors back to numpy
                out_model = cast_dict_to_numpy(out)

                ###Optional: render motion locally (commented out for speed)
                # C.update_conf({
                #     "playback_fps": 60,
                #     "auto_set_floor": True,
                #     "z_up": True,
                #     "smplx_models": "data/body_models"
                # # })
                # import shutil
                # print("=== DEMO ENVIRONMENT ===")
                # print("Python:", sys.executable)
                # print("FFmpeg:", shutil.which("ffmpeg"))
                # print("PATH:", os.environ.get("PATH"))
                # renderer = HeadlessRenderer()
                # render_motion(renderer=renderer, datum=out_model, filename='output_test.mp4', pose_repr='6d')

                # Save generated motion for analysis/debugging
                os.makedirs(MOTION_OUTPUT_DIR, exist_ok=True)
                output_path = os.path.join(MOTION_OUTPUT_DIR, f'output_motion_{ix}.npz')
                np.savez(output_path, **out_model)
                ix += 1
                print(f"[DEBUG] Saved motion to {output_path}")

                # Send motion back to Unity frame-by-frame at 30 FPS
                send_motion_to_unity(conn, out_model)

    except KeyboardInterrupt:
        print("\nServer shutting down manually...")
    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())
    finally:
        conn.close()
        server.close()
        print("Server shut down.")

if __name__ == "__main__":
    # Start the TCP server
    start_server()
