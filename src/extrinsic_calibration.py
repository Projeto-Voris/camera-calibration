import numpy as np
import cv2
import yaml
import matplotlib as plt
import os
from collections import deque

class ExtrinsicCalib:
    def __init__(self, yaml_file, markers_ids=[0, 1, 2, 3], markers_size=100, dpi=600, aruco_d: int = cv2.aruco.DICT_5X5_100):

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_d)
        if self.aruco_dict is None:
            print("Erro ao carregar o dicionário ArUco.")
        self.markers_ids = markers_ids # list of markers ids
        self.markers_size = markers_size # in millimeters
        self.dpi = dpi

        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        self.read_yaml_file(yaml_file)

        self.data = {
            'left': {},
            'right': {}
        }

        for marker_id in self.markers_ids:
            self.data['left'][marker_id] = {'rvec': deque(maxlen=30), 'tvec': deque(maxlen=30)}
            self.data['right'][marker_id] = {'rvec': deque(maxlen=30), 'tvec': deque(maxlen=30)}


    def read_yaml_file(self, yaml_file):
        """
        Read YAML file to extract cameras parameters
        """
        # Load the YAML file
        with open(yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
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


    def generat_aruco_markers(self, markers_path):
        
        os.makedirs(markers_path, exist_ok=True)
        
        a4_width, a4_height = 210, 297 # mm
        mm_to_pixels = self.dpi / 25.4
        a4_width_px, a4_height_px = int(a4_width*mm_to_pixels), int(a4_height*mm_to_pixels)
        a4_image = np.ones((int(a4_height*mm_to_pixels), int(a4_width*mm_to_pixels)), dtype=np.uint8) * 255  # White background

        marker_size_pixels = int(self.markers_size * mm_to_pixels)
        x, y = (a4_width_px - marker_size_pixels) // 2, (a4_height_px - marker_size_pixels) // 2  # Center the marker

        for marker_id in self.markers_ids:
            a4_image = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255  # White background
            marker = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, marker_size_pixels)
            marker = cv2.rotate(marker, cv2.ROTATE_180)
            
            a4_image[y:y+marker_size_pixels, x:x+marker_size_pixels] = marker

            # Save the A4 image with the marker
            cv2.imwrite(os.path.join(markers_path, f'aruco_marker_{marker_id}_a4.png'), a4_image)

    def detect_aruco_markers(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict)
        
        if ids is None:
            print("Nenhum marcador ArUco detectado.")
            return corners, ids
       
        return corners, ids

    def show_detected_markers(self, image, corners, ids):
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow('Detected markers', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_stereo_detection_markers(self, left, right, ids_left, ids_right, corners_left, corners_right):
        image_left = cv2.aruco.drawDetectedMarkers(left, corners_left, ids_left)
        image_right = cv2.aruco.drawDetectedMarkers(right, corners_right, ids_right)
        concat = np.concatenate((image_left, image_right), axis=1)
        cv2.namedWindow('stereo markers', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('stereo markers', concat.shape[1]//2, concat.shape[0]//2)
        cv2.imshow('stereo markers', concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def solve_pnp_maker(self, ids, corners, camera='left'):

        for i in range(len(ids)):
            # Ensure i is an integer
            corner = corners[i]
            
            squareLength = self.markers_size/1000  # Convert to meters

            # Define the 3D points of the marker corners in the world coordinate space
            marker_points = np.array([
                [-squareLength / 2, squareLength / 2, 0],
                [squareLength / 2, squareLength/ 2, 0],
                [squareLength / 2, -squareLength/ 2, 0],
                [-squareLength / 2, -squareLength/ 2, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                objectPoints=marker_points,
                imagePoints=corner,
                cameraMatrix=self.camera_params[camera]['kk'],
                distCoeffs=self.camera_params[camera]['kc'],
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                self.data[camera][i]['rvec'].append(rvec)
                self.data[camera][i]['tvec'].append(tvec)
