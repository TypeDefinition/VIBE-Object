# [VIBE-Object Start]

import math

'''
    Taking the smpl pose data and bone index
    Returns a quaternion 
'''
def get_bone_rotation(frame_pose, index):
    axis_angle = (frame_pose[index * 3], frame_pose[index * 3 + 1], frame_pose[index * 3 + 2])
    angle = math.sqrt(axis_angle[0] ** 2 + axis_angle[1] ** 2 + axis_angle[2] ** 2)
    if angle < 1e-5:  # Check if angle is close to zero to avoid division by zero
        return (1.0, 0.0, 0.0, 0.0)
    else:
        axis = (axis_angle[0] / angle, axis_angle[1] / angle, axis_angle[2] / angle)
        half_angle = angle / 2
        s = math.sin(half_angle)
        return (math.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s)
    
def quaternion_multiply(quat1, quat2):
    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

# Get joint index from kp_utils.get_spin_joint_names().
def get_right_wrist_translation(frame_joints32):
    return frame_joints32[4]

def get_left_wrist_translation(frame_joints32):
    return frame_joints32[7]

def get_head_translation(frame_joints32):
    return frame_joints32[43]

# Get bone index from fbx_output.bone_name_from_index. This is SMPL parameters kp_utils.get_smpl_joint_names().
RIGHT_HAND_INHERITANCE = [0, 3, 6, 9, 14, 17, 19, 21, 23]
LEFT_HAND_INHERITANCE = [0, 3, 6, 9, 13, 16, 18, 20, 22]

LEFT_LEG_INHERITANCE = [0, 1, 4, 7, 10]
RIGHT_LEG_INHERITANCE = [0, 2, 5, 8, 11]

HEAD_INHERITANCE = [0, 3, 6, 9, 12, 15]

'''
    Parameter:
        _smplJoints: get information of 3D joints in axis angle format
        _boneId: the bone we want the rotation of
    Return:
        Quaternion matrix stating the rotation of the bone
'''
def get_rotation(_smplJoints, _boneId):

    # find the parents of boneId and which part of the body it belongs to
    inheritance = RIGHT_HAND_INHERITANCE
    if _boneId in LEFT_HAND_INHERITANCE:
        inheritance = LEFT_HAND_INHERITANCE
    elif _boneId in LEFT_LEG_INHERITANCE:
        inheritance = LEFT_LEG_INHERITANCE
    elif _boneId in RIGHT_LEG_INHERITANCE:
        inheritance = RIGHT_LEG_INHERITANCE
    elif _boneId in HEAD_INHERITANCE:
        inheritance = HEAD_INHERITANCE    

    return get_rotation_from_body_part(_smplJoints, _boneId, inheritance)
    

'''
    Parameter:
        _smplJoints: get information of 3D joints in axis angle format
        _boneId: the bone we want the rotation of
        _inheritance: part of the body and the bones in that part that the boneId belongs to
    Return:
        Quaternion matrix stating the rotation of the bone
'''
def get_rotation_from_body_part(_smplJoints, _boneId, _inheritance):
    finalRotation = (1.0, 0.0, 0.0, 0.0) # identity quaternion
    for parentId in _inheritance:
        finalRotation = quaternion_multiply(finalRotation, get_bone_rotation(_smplJoints, parentId))
        if parentId == _boneId:
            break
    
    return finalRotation

# [VIBE-Object End]