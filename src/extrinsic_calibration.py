import numpy as np
import cv2
import yaml
import matplotlib as plt
import os

class ExtrinsicCalib:
    def __init__(self, yaml_file, markers_ids=[0, 1, 2, 3], markers_size=100, dpi, aruco_d: int = cv2.aruco.DICT_4X4_100):

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_d)
        self.markers_ids = markers_ids # list of markers ids
        self.markers_size = markers_size # in millimeters
        self.dpi = dpi

        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        self.read_yaml_file(yaml_file)


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
        a4_image = np.ones((np.floor(a4_height*mm_to_pixels), np.floor(a4_width*mm_to_pixels)), dtype=np.uint8) * 255  # White background

        marker_size_pixels = int(self.markers_size * mm_to_pixels)
        x, y = (a4_width - marker_size_pixels) // 2, (a4_height - marker_size_pixels) // 2  # Center the marker

        for marker_id in self.markers_ids:
            a4_image = np.ones((a4_height, a4_width), dtype=np.uint8) * 255  # White background
            marker = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, marker_size_pixels)
            marker = cv2.rotate(marker, cv2.ROTATE_180)
            
            a4_image[y:y+marker_size_pixels, x:x+marker_size_pixels] = marker

            # Save the A4 image with the marker
            cv2.imwrite(os.path.join(markers_path, f'aruco_marker_{marker_id}_a4.png'), a4_image)

    def detect_aruco_markers(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, self.aruco_dict)
        return image, corners, ids
    
    def show_detected_markers(self, image, corners, ids):
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow('Detected markers', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def solve_pnp_maker(self, camera: str,  marker_id: int, corners: tuple):
        """
        Solve the PnP problem for a single marker
        """
        # Get the marker position in meters
        objPoints = np.array([[-self.marker_size/2, -self.marker_size/2, 0],
                              [self.marker_size/2, -self.marker_size/2, 0],
                              [self.marker_size/2, self.marker_size/2, 0],
                              [-self.marker_size/2, self.marker_size/2, 0]], dtype=np.float32)*1e-3
        success, rvec, tvec = cv2.solvePnP(objPoints, corners, self.camera_params[camera]['kk'], self.camera_params[camera]['kc'])
        if success:
            rot_matrix, _ = cv2.Rodrigues(rvec)
            translation_matrix = np.eye(4)
            translation_matrix[:3, :3] = rot_matrix
            translation_matrix[:3, 3] = tvec.flatten()
            return translation_matrix
        else:
            raise Warning('Failed to solve PnP problem')
            return None
