import cdflib
import quaternion
import numpy as np


def convert_h36m_cdf_to_pos_rot(cdf: cdflib.CDF) -> list:
    poses = cdf['Pose']  # 1 x n x 96
    poses = poses.reshape((poses.shape[1], poses.shape[2] // 3, 3))  # 1 x n x 32 x 3
    print(poses.shape[0])


    def dot_last_axis(a, b):
        return np.multiply(a, b).sum(len(a.shape) - 1)

    def project_to_line(point, line_a, line_b):
        point_k = np.expand_dims(
            dot_last_axis(point - line_a, line_b - line_a) / dot_last_axis(line_b - line_a, line_b - line_a), 1)
        return line_a + point_k * (line_b - line_a)

    head_top = poses[:, 15]  # 1 x n x 3
    face = poses[:, 14]
    neck = poses[:, 13]
    head_center = project_to_line(face, neck, head_top)
    left_wrist = poses[:, 19]
    left_thumb = poses[:, 21]
    left_hand_top = poses[:, 22]
    left_hand_center = (left_wrist + left_hand_top) / 2
    right_wrist = poses[:, 27]
    right_thumb = poses[:, 29]
    right_hand_top = poses[:, 30]
    right_hand_center = (right_wrist + right_hand_top) / 2
    hip_center = poses[:, 11]
    waist_center = poses[:, 12]
    shoulder_center = poses[:, 13]
    waist_up = shoulder_center - waist_center
    hip_right = poses[:, 1] - poses[:, 6]
    shoulder_right = poses[:, 25] - poses[:, 17]  # this may be incorrect
    waist_right = hip_right * 2 + shoulder_right  # more emphasis on hip
    waist_front = np.cross(waist_up, waist_right)
    left_ankle = poses[:, 3]
    left_toe = poses[:, 5]
    left_knee = poses[:, 2]
    left_foot_center = (left_ankle + left_toe) / 2
    right_ankle = poses[:, 8]
    right_toe = poses[:, 10]
    right_knee = poses[:, 2]
    right_foot_center = (right_ankle + right_toe) / 2
    # %%
    def normalize(vec):
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)

    def look_at_quaternion_tensor(up, front):
        up = normalize(up)
        front = normalize(front)
        right = np.cross(front, up)
        up = np.cross(right, front)
        mat = np.dstack([right, front, up])
        return quaternion.from_rotation_matrix(mat)

    head_quat = look_at_quaternion_tensor(head_top - head_center, face - head_center)
    left_hand_quat = look_at_quaternion_tensor(left_thumb - left_wrist, left_hand_top - left_wrist)
    right_hand_quat = look_at_quaternion_tensor(right_thumb - right_wrist, right_hand_top - right_wrist)
    waist_quat = look_at_quaternion_tensor(waist_up, waist_front)
    left_foot_quat = look_at_quaternion_tensor(left_knee - left_ankle, left_toe - left_ankle)
    right_foot_quat = look_at_quaternion_tensor(right_knee - right_ankle, right_toe - right_ankle)

    head_info = np.concatenate((head_center / 1000, quaternion.as_float_array(np.normalized(head_quat))), axis=1)
    head_info = head_info[:, np.newaxis, :]
    left_hand_info = np.concatenate((left_hand_center / 1000, quaternion.as_float_array(np.normalized(left_hand_quat))),
                                    axis=1)
    left_hand_info = left_hand_info[:, np.newaxis, :]
    right_hand_info = np.concatenate(
        (right_hand_center / 1000, quaternion.as_float_array(np.normalized(right_hand_quat))), axis=1)
    right_hand_info = right_hand_info[:, np.newaxis, :]
    waist_info = np.concatenate((waist_center / 1000, quaternion.as_float_array(np.normalized(waist_quat))), axis=1)
    waist_info = waist_info[:, np.newaxis, :]
    left_foot_info = np.concatenate((left_foot_center / 1000, quaternion.as_float_array(np.normalized(left_foot_quat))),
                                    axis=1)
    left_foot_info = left_foot_info[:, np.newaxis, :]
    right_foot_info = np.concatenate(
        (right_foot_center / 1000, quaternion.as_float_array(np.normalized(right_foot_quat))), axis=1)
    right_foot_info = right_foot_info[:, np.newaxis, :]

    outputs = np.concatenate((head_info, left_hand_info, right_hand_info, waist_info, left_foot_info, right_foot_info),
                             axis=1)
    # print(head_info.shape, outputs.shape)

    return outputs
