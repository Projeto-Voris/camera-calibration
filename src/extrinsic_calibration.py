import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R


class ExtrinsicCalib:
    def __init__(self, marker_ids=list(range(50)), marker_size=100, dpi=600, aruco_dict=cv2.aruco.DICT_5X5_100): # Revertido: aruco_dict
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        if self.aruco_dict is None:
            raise ValueError("Erro ao carregar o dicionário ArUco.")
        
        self.marker_ids = marker_ids
        self.marker_size = marker_size
        self.dpi = dpi

        self.charuco_board = cv2.aruco.CharucoBoard(
            size=(12, 8),
            squareLength=marker_size if isinstance(marker_size, (int, float)) else float(marker_size[0]),
            markerLength=marker_size if isinstance(marker_size, (int, float)) else float(marker_size[1]),
            dictionary=self.aruco_dict
        )

        self.camera_params = {}

    def set_camera_params(self, camera, kk, kc): # Revertido: camera
        """
        Configura os parâmetros intrínsecos (kk) e de distorção (kc) para uma câmera específica.
        """
        self.camera_params[camera] = {'kk': np.array(kk, dtype=np.float64), 'kc': np.array(kc, dtype=np.float64)}

    def detect_charuco_board(self, image):
        if image is None:
            # print("Erro: Imagem de entrada é None.") # Mantido como no original (sem print) ou opcional
            return None, None
        
        gray_image = image
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) != 2:
             # print(f"Erro: Formato de imagem não suportado: {image.shape}")
             return None, None

        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, self.aruco_dict)
        
        if ids is None or len(ids) == 0:
            # print("No markers detected in the image.") # Mantido como no original
            return None, None

        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, 
            markerIds=ids, 
            image=gray_image, 
            board=self.charuco_board
        )
        
        return charuco_corners, charuco_ids
    
    def match_charuco_points(self, charuco_corners, charuco_ids): # Revertido para original
        """
        Corresponde os cantos Charuco detectados com os pontos 3D do tabuleiro.
        """
        if charuco_corners is None or charuco_ids is None or \
           len(charuco_corners) == 0 or len(charuco_ids) == 0:
            return None, None # Adicionando verificação para evitar erro se entradas forem None/vazias

        obj_points, img_points = self.charuco_board.matchImagePoints(
            detectedCorners=charuco_corners, 
            detectedIds=charuco_ids
        )
        return obj_points, img_points


    def pose_estimation_charuco(self, camera, obj_points, img_points): 
        if camera not in self.camera_params:
            print(f"Erro: Parâmetros intrínsecos não definidos para a câmera {camera}.")
            return None
            
        if obj_points is None or img_points is None or len(obj_points) < 4 or len(img_points) < 4:
            # print(f"Skipping pose estimation for camera {camera} due to insufficient points.")
            return None

        kk = self.camera_params[camera]['kk']
        kc = self.camera_params[camera]['kc']

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=kk,
            distCoeffs=kc,
            flags=cv2.SOLVEPNP_IPPE 
        )
        if success:
            T = np.eye(4) 
            R, _ = cv2.Rodrigues(rvec)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            return T
        return None

    def calculate_relative_pose(self, T_0_2, T_1_2):
        if T_0_2 is None or T_1_2 is None:
            return None
        try:
            return T_0_2 @ np.linalg.inv(T_1_2) 
        except np.linalg.LinAlgError:
            # print("Erro: Singularidade ao inverter matriz de transformação.")
            return None
    
    def filter_images_by_precision(self, transformations, mad_multiplier=3.0):
        """
        Filtra imagens com base na consistência das transformações relativas (translações),
        usando o Desvio Absoluto Mediano (MAD).

        :param transformations: Lista ou array NumPy de translações (Nx3).
        :param mad_multiplier: Fator multiplicativo para o MAD.
        :return: Índices das imagens que atendem ao critério de consistência.
        """

        if not isinstance(transformations, (list, np.ndarray)) or len(transformations) == 0:
            return []
        
        translations_np = np.array(transformations)
        if translations_np.ndim == 1: 
            translations_np = translations_np.reshape(1, -1)

        if translations_np.shape[0] <= 2: # MAD não é muito útil para 1 ou 2 amostras
            return list(range(translations_np.shape[0]))

        num_translations, num_axes = translations_np.shape
        if num_axes != 3:
            # print(f"Aviso: 'transformations' não é Nx3 (formato: {translations_np.shape}).") # Opcional
            raise ValueError(f"Transformations should be an Nx3 array, but got shape {translations_np.shape}.")


        final_valid_mask = np.ones(num_translations, dtype=bool)
        for axis_idx in range(num_axes): 
            axis_data = translations_np[:, axis_idx]
            
            median_val = np.median(axis_data)
            abs_deviations = np.abs(axis_data - median_val)
            mad_val = np.median(abs_deviations)

            if mad_val < 1e-9: 
                axis_mask_component = (abs_deviations <= 1e-9) 
            else:
                current_mad_multiplier_value = mad_multiplier * mad_val 
                axis_mask_component = (abs_deviations <= current_mad_multiplier_value)
            
            final_valid_mask = np.logical_and(final_valid_mask, axis_mask_component)
            
        return np.where(final_valid_mask)[0].tolist()
    
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