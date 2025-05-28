import numpy as np
import cv2
import cv2.aruco as aruco
import yaml
import matplotlib.pyplot as plt
from collections import deque

class ExtrinsicCalib:
    def __init__(self, marker_ids=list(range(50)), marker_size=100, dpi=600, aruco_dict=cv2.aruco.DICT_5X5_100):
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

        # Inicializa os parâmetros da câmera como vazios
        self.camera_params = {}

    def set_camera_params(self, camera, kk, kc):
        """
        Configura os parâmetros intrínsecos (kk) e de distorção (kc) para uma câmera específica.
        """
        self.camera_params[camera] = {'kk': np.array(kk), 'kc': np.array(kc)}

    def detect_charuco_board(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict)
        if corners is None or len(corners) == 0 or ids is None or len(ids) == 0:
            print("No markers detected in the image.")
            return None, None

        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=image, board=self.charuco_board
        )
        return charuco_corners, charuco_ids
    
    def match_charuco_points(self, charuco_corners, charuco_ids):
        obj_points, img_points = self.charuco_board.matchImagePoints(detectedCorners=charuco_corners, detectedIds=charuco_ids)
        return obj_points, img_points

    def pose_estimation_charuco(self, camera, obj_points, img_points):

        if len(obj_points) < 4 or len(img_points) < 4:
            print(f"Skipping pose estimation for camera {camera} due to insufficient points.")
            return None

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
    
    def filter_images_by_precision(self, transformations, threshold=1.0):
        """
        Filtra imagens com base na precisão das transformações relativas.
        
        :param transformations: Lista de transformações relativas (Nx3 array).
        :param threshold: Limite máximo de amplitude permitido para cada eixo.
        :return: Índices das imagens que atendem ao critério de precisão.
        """
        transformations = np.array(transformations)
        peak_to_peak = np.ptp(transformations, axis=0)  # Calcula a amplitude para cada eixo (x, y, z)
        
        if np.all(peak_to_peak <= threshold):
            return list(range(len(transformations)))  # Todas as imagens atendem ao critério
        else:
            # Filtrar imagens com base na variação em cada eixo
            valid_indices = [
                i for i, t in enumerate(transformations)
                if np.all(np.abs(t - np.mean(transformations, axis=0)) <= threshold)
            ]
            return valid_indices