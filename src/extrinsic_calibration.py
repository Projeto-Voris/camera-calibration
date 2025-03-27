import numpy as np
import cv2
import yaml
import matplotlib as plt
import os
import matplotlib.pyplot as plt
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
            self.data['left'][marker_id] = {'translation_matrix': deque(maxlen=30)}
            self.data['right'][marker_id] = {'translation_matrix': deque(maxlen=30)}

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


    def pose_estimation (self, ids, corners, camera: str,image, show=True ):

        for i in range(len(ids)):
            # Ensure i is an integer
            corner = corners[i]
            
            squareLength = self.markers_size/1000 # Convert to meters

            # Define the 3D points of the marker corners in the world coordinate space
            marker_points = np.array([
                [-squareLength / 2, squareLength / 2, 0],
                [squareLength / 2, squareLength/ 2, 0],
                [squareLength / 2, -squareLength/ 2, 0],
                [-squareLength / 2, -squareLength/ 2, 0]
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                objectPoints=marker_points,
                imagePoints=corner,
                cameraMatrix=self.camera_params[camera]['kk'],
                distCoeffs=self.camera_params[camera]['kc'],
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if show:
                image = cv2.drawFrameAxes(
                    image=image, 
                    cameraMatrix=self.camera_params[camera]['kk'],
                    distCoeffs=self.camera_params[camera]['kc'], 
                    rvec=rvec, 
                    tvec=tvec, 
                    length=squareLength/2,
                    thickness=2)

            if success:

                self.data[camera].setdefault(i, {'rvec': [], 'tvec': []})
                rot_matrix, _ = cv2.Rodrigues(rvec)
                translation_matrix = np.eye(4)
                translation_matrix[:3, :3] = rot_matrix
                translation_matrix[:3, 3] = tvec.flatten()

                if 'translation_matrix' not in self.data[camera][i]:
                    self.data[camera][i]['translation_matrix'] = []
                    
                self.data[camera][i]['translation_matrix'].append(translation_matrix)

        if show:
                cv2.imshow('Detected markers', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def calculate_relative_pose(self, T_left, T_right):
        """
        Calculate the relative pose between two cameras given their transformation matrices.
        """
        # Calculate the relative transformation: T_right_to_left = T_left^-1 * T_right
        T_left_inv = np.linalg.inv(T_left)  # Invert the transformation matrix of the left camera
        T_right_to_left = T_right  @ T_left_inv  # Compute the relative transformation (right to left)
        
        R_relative = np.eye(3) # Extract the rotation matrix from the relative transformation
        t_relative = T_right_to_left[:3, 3]  # Extract the translation vector from the relative transformation

        print(f"R_relative:\n{R_relative}")
        print(f"T_relative:\n{t_relative}")

        return R_relative, t_relative

    def plot_camera_axes(self, R, t, ax, label):
        # Define os eixos da câmera no sistema de coordenadas local
        axes_length = self.markers_size/1000  # Comprimento dos eixos ajustado pela escala
        origin = t.flatten()
        x_axis = origin + R[:, 0] * axes_length
        y_axis = origin + R[:, 1] * axes_length
        z_axis = origin + R[:, 2] * axes_length

        # Plota os eixos
        ax.quiver(*origin, *(x_axis - origin), color='r', label=f'{label} X-axis')
        ax.quiver(*origin, *(y_axis - origin), color='g', label=f'{label} Y-axis')
        ax.quiver(*origin, *(z_axis - origin), color='b', label=f'{label} Z-axis')

    def visualize_relative_pose(self, R_relative, t_relative):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Origem da câmera esquerda
        self.plot_camera_axes(np.eye(3), np.zeros((3, 1)), ax, label='Left Camera')

        # Origem da câmera direita (relativa à esquerda)
        self.plot_camera_axes(R_relative, t_relative, ax, label='Right Camera')

        # Configurações do gráfico
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Relative Pose of Cameras')
        ax.legend()

        # Ajustar os limites dos eixos para aumentar o tamanho do mapa 3D
        ax.set_xlim([-2, 1])  # Ajuste os valores conforme necessário
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        plt.show()
