import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from collections import deque

class ExtrinsicCalib:
    def __init__(self, yaml_file, marker_ids=list(range(50)), marker_size=100, dpi=600, aruco_dict=cv2.aruco.DICT_5X5_100):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        if self.aruco_dict is None:
            raise ValueError("Erro ao carregar o dicionário ArUco.")
        
        self.marker_ids = marker_ids
        self.marker_size = marker_size
        self.dpi = dpi

        self.charuco_board = cv2.aruco.CharucoBoard(
            size=(12, 8),
            squareLength=marker_size if isinstance(marker_size, int) else marker_size[0],
            markerLength=marker_size if isinstance(marker_size, int) else marker_size[1],
            dictionary=self.aruco_dict
        )

        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        self.read_yaml_file(yaml_file)

        self.data = {
            'left': {marker_id: {'translation_matrix': deque(maxlen=30)} for marker_id in self.marker_ids},
            'right': {marker_id: {'translation_matrix': deque(maxlen=30)} for marker_id in self.marker_ids}
        }

    def read_yaml_file(self, yaml_file):
        with open(yaml_file) as file:
            params = yaml.safe_load(file)

        self.camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        self.camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        self.camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        self.camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)

        self.camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        self.camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        self.camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        self.camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

        self.camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
        self.camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

    def detect_charuco_board(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict)
        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=image, board=self.charuco_board
        )
        return charuco_corners, charuco_ids if charuco_ids is not None else (None, None)
    
    def match_charuco_points(self, charuco_corners, charuco_ids):
        """
        Match the detected Charuco corners with their corresponding ids.
        """
        obj_points, img_points = self.charuco_board.matchImagePoints(detectedCorners=charuco_corners, detectedIds=charuco_ids)
        return obj_points, img_points

    def pose_estimation_charuco(self, camera, obj_points, img_points):
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=self.camera_params[camera]['kk'],
            distCoeffs=self.camera_params[camera]['kc'],
            flags=cv2.SOLVEPNP_IPPE
        )
        if success:
            T = np.eye(4)
            T[:3, :3], _ = cv2.Rodrigues(rvec)
            T[:3, 3] = tvec.flatten()
            return T
        return None

    def calculate_relative_pose(self, T_0_2, T_1_2):
        return T_0_2 @ np.linalg.inv(T_1_2)

    def transform_precision(self, transforms):
        if not all(transforms):
            return None

        positions = np.array([transform[:3, 3] for transform in transforms])
        rotations = [cv2.Rodrigues(transform[:3, :3])[0] for transform in transforms]

        pos_range = np.ptp(positions, axis=0)
        rot_range = np.ptp(rotations, axis=0)

        if np.mean(pos_range) > 0.6:
            print("Transformações não são precisas o suficiente.")
            return None

        translation_matrix = np.eye(4)
        translation_matrix[:3, :3] = cv2.Rodrigues(np.mean(rotations, axis=0))[0]
        translation_matrix[:3, 3] = np.mean(positions, axis=0).flatten()
        return translation_matrix

    def get_valid_markers(self):
        valid_markers = [
            marker_id
            for marker_id in self.markers_ids
            if self.data['left'][marker_id]['translation_matrix']
               and self.data['right'][marker_id]['translation_matrix']
        ]

        # print(valid_markers)
        return valid_markers