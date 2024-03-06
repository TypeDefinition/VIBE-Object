# [VIBE-Object Start]
# Just whatever chapalang stuff I need to get shit to work.

import math
from mathutils import Vector, Quaternion

def get_bone_rotation(frame_pose, index):
    axis_angle = Vector((frame_pose[index * 3], frame_pose[index * 3 + 1], frame_pose[index * 3 + 2]))
    angle = axis_angle.length
    axis = axis_angle.normalized()
    return Quaternion((axis.x, axis.y, axis.z), angle)

# Get joint index from kp_utils.get_spin_joint_names().
def get_right_wrist_translation(frame_joints32):
    return frame_joints32[4]

def get_left_wrist_translation(frame_joints32):
    return frame_joints32[7]

def get_head_translation(frame_joints32):
    return frame_joints32[43]

# Get bone index from fbx_output.bone_name_from_index.
def get_right_wrist_rotation(frame_pose):
    return (get_bone_rotation(frame_pose, 0) @
            get_bone_rotation(frame_pose, 3) @
            get_bone_rotation(frame_pose, 6) @
            get_bone_rotation(frame_pose, 9) @
            get_bone_rotation(frame_pose, 14) @
            get_bone_rotation(frame_pose, 17) @
            get_bone_rotation(frame_pose, 19) @
            get_bone_rotation(frame_pose, 21))

def get_right_hand_rotation(frame_pose):
    return (get_bone_rotation(frame_pose, 0) @
            get_bone_rotation(frame_pose, 3) @
            get_bone_rotation(frame_pose, 6) @
            get_bone_rotation(frame_pose, 9) @
            get_bone_rotation(frame_pose, 14) @
            get_bone_rotation(frame_pose, 17) @
            get_bone_rotation(frame_pose, 19) @
            get_bone_rotation(frame_pose, 21) @
            get_bone_rotation(frame_pose, 23))

def get_left_wrist_rotation(frame_pose):
    return (get_bone_rotation(frame_pose, 0) @
            get_bone_rotation(frame_pose, 3) @
            get_bone_rotation(frame_pose, 6) @
            get_bone_rotation(frame_pose, 9) @
            get_bone_rotation(frame_pose, 13) @
            get_bone_rotation(frame_pose, 16) @
            get_bone_rotation(frame_pose, 18) @
            get_bone_rotation(frame_pose, 20))

def get_left_hand_rotation(frame_pose):
    return (get_bone_rotation(frame_pose, 0) @
            get_bone_rotation(frame_pose, 3) @
            get_bone_rotation(frame_pose, 6) @
            get_bone_rotation(frame_pose, 9) @
            get_bone_rotation(frame_pose, 13) @
            get_bone_rotation(frame_pose, 16) @
            get_bone_rotation(frame_pose, 18) @
            get_bone_rotation(frame_pose, 20) @
            get_bone_rotation(frame_pose, 22))

def get_head_rotation(frame_pose):
    return (get_bone_rotation(frame_pose, 0) @
            get_bone_rotation(frame_pose, 3) @
            get_bone_rotation(frame_pose, 6) @
            get_bone_rotation(frame_pose, 9) @
            get_bone_rotation(frame_pose, 12) @
            get_bone_rotation(frame_pose, 15))
# [VIBE-Object End]