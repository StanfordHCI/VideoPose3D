import cdflib
import quaternion
import numpy as np
'''
def cal_pos_rot(poses: np.ndarray) -> (np.ndarray):
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
    shoulder_center = poses[:, 13]
    waist_real_center = (hip_center + shoulder_center) / 2
    waist_up = shoulder_center - waist_real_center
    waist_center = (waist_real_center + hip_center) / 2
    left_hip = poses[:, 6]
    right_hip = poses[:, 1]
    hip_right = right_hip - left_hip
    left_shoulder = poses[:, 17]
    right_shoulder = poses[:, 25]
    shoulder_right = right_shoulder - left_shoulder  # this may be incorrect
    waist_right = hip_right * 2 + shoulder_right  # more emphasis on hip
    waist_front = np.cross(waist_up, waist_right)
    left_ankle = poses[:, 8]
    left_toe = poses[:, 10]
    left_knee = poses[:, 7]
    left_foot_center = (left_ankle + left_toe) / 2
    right_ankle = poses[:, 3]
    right_toe = poses[:, 5]
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
    outputs = np.concatenate((head_info, left_hand_info, right_hand_info), axis=1)
    # print(head_info.shape, outputs.shape)
    return outputs
'''

def cal_pos_rot(poses: np.ndarray) -> np.ndarray:
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
    shoulder_center = poses[:, 13]
    waist_real_center = (hip_center + shoulder_center) / 2
    waist_up = shoulder_center - waist_real_center
    waist_center = (waist_real_center + hip_center) / 2
    hip_right = poses[:, 1] - poses[:, 6]
    shoulder_right = poses[:, 25] - poses[:, 17]  # this may be incorrect
    waist_right = hip_right * 2 + shoulder_right  # more emphasis on hip
    waist_front = np.cross(waist_up, waist_right)
    left_ankle = poses[:, 8]
    left_toe = poses[:, 10]
    left_knee = poses[:, 7]
    left_foot_center = (left_ankle + left_toe) / 2
    right_ankle = poses[:, 3]
    right_toe = poses[:, 5]
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

    outputs = np.concatenate((head_info, left_hand_info, right_hand_info),
                             axis=1)
    # print(head_info.shape, outputs.shape)
    return outputs


def convert_h36m_cdf_to_pos_rot(cdf: cdflib.CDF) -> (np.ndarray, np.ndarray):
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
    shoulder_center = poses[:, 13]
    waist_real_center = (hip_center + shoulder_center) / 2
    waist_up = shoulder_center - waist_real_center
    waist_center = (waist_real_center + hip_center) / 2
    left_hip = poses[:, 6]
    right_hip = poses[:, 1]
    hip_right = right_hip - left_hip
    left_shoulder = poses[:, 17]
    right_shoulder = poses[:, 25]
    shoulder_right = right_shoulder - left_shoulder  # this may be incorrect
    waist_right = hip_right * 2 + shoulder_right  # more emphasis on hip
    waist_front = np.cross(waist_up, waist_right)
    left_ankle = poses[:, 8]
    left_toe = poses[:, 10]
    left_knee = poses[:, 7]
    left_foot_center = (left_ankle + left_toe) / 2
    right_ankle = poses[:, 3]
    right_toe = poses[:, 5]
    right_knee = poses[:, 2]
    right_foot_center = (right_ankle + right_toe) / 2

    def vec_len(vec):
        length = np.linalg.norm(vec, axis=1, keepdims=True)
        return np.mean(length)

    # Model v2.1 body metric
    nose_neck = vec_len(head_center - shoulder_center)
    shoulder_length = vec_len(shoulder_right)
    left_elbow = poses[:, 18]
    left_shoulder_elbow = vec_len(left_shoulder - left_elbow)
    left_elbow_wrist = vec_len(left_elbow - left_wrist)
    right_elbow = poses[:, 26]
    right_shoulder_elbow = vec_len(right_shoulder - right_elbow)
    right_elbow_wrist = vec_len(right_elbow - right_wrist)
    shoulder_hip = vec_len(shoulder_center - hip_center)
    hip_length = vec_len(hip_right)
    left_hip_knee = vec_len(left_hip - left_knee)
    left_knee_ankle = vec_len(left_knee - left_ankle)
    left_ankle_foot = vec_len(left_ankle - left_foot_center)
    right_hip_knee = vec_len(right_hip - right_knee)
    right_knee_ankle = vec_len(right_knee - right_ankle)
    right_ankle_foot = vec_len(right_ankle - right_foot_center)
    output_lengths = [nose_neck, shoulder_length, shoulder_hip, hip_length,
                      left_shoulder_elbow, right_shoulder_elbow,
                      left_elbow_wrist, right_elbow_wrist,
                      left_hip_knee, right_hip_knee,
                      left_knee_ankle, right_knee_ankle,
                      left_ankle_foot, right_ankle_foot]

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
    return outputs, output_lengths


def visual_test(pos_3d):
    import matplotlib.pyplot as plt
    import cv2
    # vr_poses = [[head_center, head_quat], [left_hand_center, left_hand_quat], [right_hand_center, right_hand_quat],
    # [waist_center, waist_quat], [left_foot_center, left_foot_quat], [right_foot_center, right_foot_quat]]
    bone_pair = [[0, 1], [0, 2], [0, 3], [3, 4], [3, 5]]

    def draw_joint(pos: np.ndarray, quat: quaternion.quaternion):
        front_vec = pos + quaternion.rotate_vectors(quat, [0, 0.2, 0])
        right_vec = pos + quaternion.rotate_vectors(quat, [0.2, 0, 0])
        up_vec = pos + quaternion.rotate_vectors(quat, [0, 0, 0.2])
        plt.plot(*zip(pos, front_vec), zdir='z', c='blue')
        plt.plot(*zip(pos, up_vec), zdir='z', c='green')
        plt.plot(*zip(pos, right_vec), zdir='z', c='red')

    def draw_bone(pos_a: np.ndarray, pos_b: np.ndarray, c='black'):
        plt.plot(*zip(pos_a, pos_b), zdir='z', c=c)

    parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
               16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]
    plt.figure(dpi=80)
    print(pos_3d.shape)
    for i in range(pos_3d.shape[0]):
        plt.clf()
        plt.subplot(111, projection='3d')
        plt.xlim(-1.700, 1.700)
        plt.ylim(-1.700, 1.700)
        plt.gca().set_zlim(0, 6.400)
        # temp_pose = pos_3d[i,:,:]
        for j in range(pos_3d.shape[1]):
            pose = pos_3d[i, j, :]
            # print(pose)
            draw_joint(pose[0:3], quaternion.as_quat_array(pose[3:7]))
        for j, k in bone_pair:
            draw_bone(pos_3d[i, j, 0:3], pos_3d[i, k, 0:3])
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        plt.gcf().canvas.draw()
        image = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        cv2.imshow('im', image)
        cv2.waitKey(1)


if __name__ == "__main__":
    path = 'data_3d_h36m_new.npz'
    data = np.load(path, allow_pickle=True)['pos_rot'].item()
    visual_test(data["S1"]["Posing 1"])
