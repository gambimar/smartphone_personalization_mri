from SKEL.skel import kin_skel
from scipy.spatial.transform import Rotation as R
import numpy as np
from src.bodyscan import segment

fk_model = {
    "pelvis": {
        "parent": [0, 0, 0],
        "rotation": ["pelvis_tilt", "pelvis_rotation", "pelvis_list"],
        "dir": [-1, 1, 1]
    },
    "femur": {
        "parent": "pelvis",
        "rotation": ["hip_flexion", "hip_rotation", "hip_adduction"],
        "dir_l": [-1, -1, -1],
        "dir_r": [-1, 1, 1]
    },
    "tibia": {
        "parent": "femur",
        "rotation": ["knee_angle", 0, 0]
    },
    "calcn": {
        "parent": "tibia",
        "rotation": ["ankle_angle", "subtalar_angle", 0],
        "dir_r": [-1, 1, 1],
        "dir_l": [-1, -1, 1],
    },
    "toes": {
        "parent": "calcn",
        "rotation": ["mtp_angle", 0, 0]
    },
    "lumbar_body": {
        "parent": "pelvis",
        "rotation": [["lumbar_extension", "thorax_extension"], ["lumbar_twist", "thorax_twist"],
                     ["lumbar_bending", "thorax_bending"], ],
        "order": "xyz",
        "dir": [-1, 1, 1]
    },
    "shoulder": {
        "parent": "lumbar_body",
        "rotation": ["scapula_abduction", "scapula_elevation", "scapula_upward_rot"],
        "dir_r": [1, 1, -1],
        "dir_l": [1, -1, 1],
        "order": "yxz"
    },
    "humerus": {
        "parent": "shoulder",
        "rotation": ['shoulder__y', 'shoulder__z', 'shoulder__x'],
        "dir_r": [1, 1, 1],
        "dir_l": [1, -1, -1],
        "order": "xyz"
    },
    "ulna": {
        "parent": "humerus",
        "rotation": ["pro_sup", "elbow_flexion", 0],
        "dir_r": [1, 1, 1],
        "dir_l": [1, -1, -1]
    },
    "hand": {
        "parent": "ulna",
        "rotation": [0, "wrist_deviation", "wrist_flexion"],
        "dir_r": [1, -1, 1],
        "dir_l": [1, 1, -1],
    },
    "thorax": {  # Only for debugging purposes
        "parent": "lumbar_body",
        "rotation": [0, 0, 0]
    },
}


def parse_list(l, pose, side=''):
    l = l.copy()
    for i in range(len(l)):
        if isinstance(l[i], list):
            l[i] = np.sum([pose[kin_skel.pose_param_names.index(j)] for j in l[i]])
        elif isinstance(l[i], str):
            if l[i][-3:-1] == '__':
                l[i] = l[i][:-3] + side + l[i][-2:]
            elif side in ["_r", "_l"]:
                l[i] = l[i] + side
            l[i] = pose[kin_skel.pose_param_names.index(l[i])].copy()
    return l


def rotate(orient_0, rotation, order):
    r = R.from_euler(order, rotation, degrees=False)
    return orient_0.copy() @ r.as_matrix()


orientation_bodies = ["pelvis",
                      "femur_r", "femur_l", "tibia_r", "tibia_l", "calcn_r", "calcn_l", "toes_r", "toes_l",
                      "lumbar_body", "shoulder_r", "shoulder_l", "humerus_r", "humerus_l", "ulna_r", "ulna_l", "hand_r",
                      "hand_l", "thorax"]

t_pose_def_joints = {
    # "humerus_r": [-np.pi/2,np.pi/2,0],
    # "humerus_l": [np.pi/2,0,0],
    # "calcn_r": [-np.pi/2, 0, 0],
    # "calcn_l": [-np.pi/2, 0, 0],
    # "torso": [-np.pi, 0, 0], # Other direction of build
}


def get_bone_orientation(pose):
    pose = np.squeeze(pose)  # To make sure that it is a 46 element array
    pose = np.array(pose)
    assert len(pose) == 46, "Pose should be 46 elements"
    orientations = {}
    for body in orientation_bodies:
        if body.endswith("_r") or body.endswith("_l"):
            side = body[-2:]
            body = body[:-2]
        else:
            side = ''

        if body == "pelvis":
            orient_0 = np.eye(3)
            orient_0 = rotate(orient_0, fk_model['pelvis']['parent'], "xyz")  # Rotate pelvis rent
            parent = None
        else:
            parent = fk_model[body]['parent']
            if parent not in orientation_bodies:
                parent = parent + side
            orient_0 = orientations[parent]

        # Check if order is specified in the segment
        if 'order' in fk_model[body]:
            order = fk_model[body]['order']
        else:
            order = "xyz"

        if body + side in t_pose_def_joints:
            rotations = t_pose_def_joints[body + side]
            orient_0 = rotate(orient_0, rotations, order)  # Rotate to t-pose

        rotations = parse_list(fk_model[body]['rotation'], pose, side)
        if "dir" in fk_model[body]:
            rotations = rotations * np.array(fk_model[body]['dir'])
        elif "dir_r" in fk_model[body] and side == "_r":
            rotations = rotations * np.array(fk_model[body]['dir_r'])
        elif "dir_l" in fk_model[body] and side == "_l":
            rotations = rotations * np.array(fk_model[body]['dir_l'])
        orientations[body + side] = rotate(orient_0, rotations, order)
    return orientations