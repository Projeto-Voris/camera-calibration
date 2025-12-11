import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import yaml
from scipy.spatial.transform import Rotation as R
from collections import deque

np.set_printoptions(precision=3, suppress=True, linewidth=120)
class ExtrinsicTransform:
    def __init__(self, yaml_path_1, yaml_path_2):
        self.active = self.read_yaml_file(yaml_path_1)
        self.passive = self.read_yaml_file(yaml_path_2)
        

    def read_yaml_file(self, yaml_file):
        """
        Read YAML file to extract cameras parameters
        """
        camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        # Load the YAML file
        with open(yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
        camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)

        camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

        camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
        camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

        return camera_params
    
    def read_transforms(self, transform_file):
        """
        Read multiple 4x4 transformation matrices from a file.
        Each matrix should be separated by a blank line.
        """
        if not os.path.exists(transform_file):
            raise FileNotFoundError(f"Transform file {transform_file} does not exist.")

        matrices = deque()
        with open(transform_file, 'r') as f:
            lines = []
            for line in f:
                if line.strip() == "":
                    if lines:
                        matrix = np.loadtxt(lines, delimiter=',')
                        if matrix.shape != (4, 4):
                            raise ValueError("Each transform matrix must be a 4x4 matrix.")
                        matrices.append(matrix)
                        lines = []
                else:
                    lines.append(line)
            # Handle last matrix if file doesn't end with blank line
            if lines:
                matrix = np.loadtxt(lines, delimiter=',')
                if matrix.shape != (4, 4):
                    raise ValueError("Each transform matrix must be a 4x4 matrix.")
                matrices.append(matrix)
        return matrices
    
    def calculate_relative_pose(self, T_0_2, T_1_2):
        T01 = deque()
        if T_0_2 is None or T_1_2 is None:
            return None
        if len(T_0_2) != len(T_1_2):
            raise ValueError("List of Transformation matrices must have the same dimensions.")
        for T02, T12 in zip(T_0_2, T_1_2):
            if T02.shape != (4, 4) or T12.shape != (4, 4):
                raise ValueError("Each transformation matrix must be a 4x4 matrix.")
            try:
                T01.append(T02 @ np.linalg.inv(T12))
            except np.linalg.LinAlgError:
                print("Erro: Singularidade ao inverter matriz de transformação.")
        return T01        
    def average_transform_svd(self, transform_list: list) -> np.ndarray:
        """
        Average homogeneous transforms using SVD for rotation and mean for translation.
        transform_list: list of 4x4 transformation matrices
        """
        if len(transform_list) == 0:
            raise ValueError("No transforms provided.")

        # Average rotation
        M = np.zeros((3, 3))
        for T_i in transform_list:
            M += T_i[:3, :3]
        M /= len(transform_list)

        U, _, Vt = np.linalg.svd(M)
        R_avg = U @ Vt
        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1
            R_avg = U @ Vt

        # Average translation
        t_sum = np.zeros(3)
        for T_i in transform_list:
            t_sum += T_i[:3, 3]
        t_avg = t_sum / len(transform_list)

        # Rebuild homogeneous matrix
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg
        return T_avg

    def average_transform_quaternion(self, transform_list: list) -> np.ndarray:
        """
        Average homogeneous transforms using quaternion mean for rotation and mean for translation.
        transform_list: list of 4x4 transformation matrices
        """
        if len(transform_list) == 0:
            raise ValueError("No transforms provided.")

        quats = []
        t_sum = np.zeros(3)

        for T_i in transform_list:
            R_i = T_i[:3, :3]
            q = R.from_matrix(R_i).as_quat()  # xyzw
            if len(quats) > 0 and np.dot(quats[0], q) < 0:
                q = -q
            quats.append(q)
            t_sum += T_i[:3, 3]

        quats = np.array(quats)
        q_mean = np.mean(quats, axis=0)
        q_mean /= np.linalg.norm(q_mean)

        R_avg = R.from_quat(q_mean).as_matrix()
        t_avg = t_sum / len(transform_list)

        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg
        return T_avg
    
def main():
    # Example usage
    active = './cfg/system2.yaml' 
    passive = './cfg/system1.yaml'
    transform_path = './data/20251211'
    transform_files = sorted(os.listdir(transform_path))
    transform = ExtrinsicTransform(active, passive)
    transform_list = {
                        'active': {'cam1': deque(), 'cam2': deque()},
                        'passive': {'cam1': deque(), 'cam2': deque()},
                        'fpv': {'cam': deque()}
                    }
    for file in transform_files:
        if file.endswith('.txt'):
            system = file.split('_')[1]
            cam = file.split('_')[2].split('.')[0]
            transform_list[system][cam] = transform.read_transforms(os.path.join(transform_path, file))


                       
    t_cam1_cam1 = transform.calculate_relative_pose(transform_list['passive']['cam1'], transform_list['passive']['cam2'])
    T_avg_svd = transform.average_transform_svd(t_cam1_cam1)
    print(f"Average Transform SVD Passive Left-Right:\n{T_avg_svd}")
    print(f"Average translate vector SVD Passive Left-Right:\n{T_avg_svd[:3, 3]}")             
    print('-' * 50)
    # t_cam1_cam1 = transform.calculate_relative_pose(transform_list['active']['cam1'], transform_list['active']['cam2'])
    # T_avg_svd = transform.average_transform_svd(t_cam1_cam1)
    # print(f"Average Transform SVD Active Left-Rightt:\n{T_avg_svd}")
    # print(f"Average translate vector SVD Active Left-Right:\n{T_avg_svd[:3, 3]}")     
    # print('-' * 50)

    # t_cam1_cam1 = transform.calculate_relative_pose(transform_list['passive']['cam1'], transform_list['active']['cam1'])
    # T_avg_svd = transform.average_transform_svd(t_cam1_cam1)
    # print(f"Average Transform SVD Left-Left:\n{T_avg_svd}")
    # print(f"Average translate vector SVD Left-Left:\n{T_avg_svd[:3, 3]}")
    # print('-' * 50)
        
    # t_cam2_cam2 = transform.calculate_relative_pose(transform_list['passive']['cam2'], transform_list['active']['cam2'])
    # T_avg_svd = transform.average_transform_svd(t_cam2_cam2)
    # print(f"Average Transform SVD Right-Right:\n{T_avg_svd}")
    # print(f"Average translate vector SVD Right-Right:\n{T_avg_svd[:3, 3]}")

    t_fpv_cam1 = transform.calculate_relative_pose(transform_list['fpv']['cam'], transform_list['passive']['cam1'])
    T_avg_svd = transform.average_transform_svd(t_fpv_cam1)
    print(f"Average Transform SVD FPV-Passive Left:\n{T_avg_svd}")
    print(f"Average translate vector SVD FPV-Passive Left:\n{T_avg_svd[:3, 3]}")     
    print('-' * 50)

    t_fpv_cam2 = transform.calculate_relative_pose(transform_list['fpv']['cam'], transform_list['passive']['cam2'])
    T_avg_svd = transform.average_transform_svd(t_fpv_cam2)
    print(f"Average Transform SVD FPV-Passive Right:\n{T_avg_svd}")
    print(f"Average translate vector SVD FPV-Passive Right:\n{T_avg_svd[:3, 3]}")     
    print('-' * 50)

if __name__ == "__main__":
    main()