import numpy as np
import cv2
import yaml
import matplotlib as plt
import os
import matplotlib.pyplot as plt
from collections import deque

class ExtrinsicCalib:
    def __init__(self, yaml_file, markers_ids=list(range(50)), markers_size=100, dpi=600,
                 aruco_d: int = cv2.aruco.DICT_4X4_50):

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_d)
        if self.aruco_dict is None:
            print("Erro ao carregar o dicionário ArUco.")
        self.markers_ids = markers_ids  # list of markers ids
        self.markers_size = markers_size  # in millimeters
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

    def generate_aruco_markers(self, markers_path):

        os.makedirs(markers_path, exist_ok=True)

        a4_width, a4_height = 210, 297  # mm
        mm_to_pixels = self.dpi / 25.4
        a4_width_px, a4_height_px = int(a4_width * mm_to_pixels), int(a4_height * mm_to_pixels)
        a4_image = np.ones((int(a4_height * mm_to_pixels), int(a4_width * mm_to_pixels)),
                           dtype=np.uint8) * 255  # White background

        marker_size_pixels = int(self.markers_size * mm_to_pixels)
        x, y = (a4_width_px - marker_size_pixels) // 2, (a4_height_px - marker_size_pixels) // 2  # Center the marker

        for marker_id in self.markers_ids:
            a4_image = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255  # White background
            marker = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, marker_size_pixels)
            marker = cv2.rotate(marker, cv2.ROTATE_180)

            a4_image[y:y + marker_size_pixels, x:x + marker_size_pixels] = marker

            # Save the A4 image with the marker
            cv2.imwrite(os.path.join(markers_path, f'aruco_marker_{marker_id}_a4.png'), a4_image)

    def detect_aruco_markers(self, image):
        if len(image.shape) == 3:
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

    def get_valid_markers(self):
        valid_markers = [
            marker_id
            for marker_id in self.markers_ids
            if self.data['left'][marker_id]['translation_matrix']
               and self.data['right'][marker_id]['translation_matrix']
        ]

        # print(valid_markers)
        return valid_markers

    def pose_estimation(self, ids, corners, camera: str, image, show=True):
        
        # print(f"id: {ids} ")
        for i, id in enumerate(ids):
            corner = corners[i]
            # print(f"Detected marker id {id}")
            marker_id = id.flatten()[0]
            
            # Define the 3D points of the marker corners in the world coordinate space
            marker_points = np.array([
                [-self.markers_size / 2, self.markers_size / 2, 0],
                [self.markers_size / 2, self.markers_size / 2, 0],
                [self.markers_size / 2, -self.markers_size / 2, 0],
                [-self.markers_size / 2, -self.markers_size / 2, 0]
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                objectPoints=marker_points,
                imagePoints=corner,
                cameraMatrix=self.camera_params[camera]['kk'],
                distCoeffs=self.camera_params[camera]['kc'],
                flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if show:
                image = cv2.drawFrameAxes(
                    image=image,
                    cameraMatrix=self.camera_params[camera]['kk'],
                    distCoeffs=self.camera_params[camera]['kc'],
                    rvec=rvec,
                    tvec=tvec,
                    length=self.markers_size / 2,
                    thickness=2)

            if success:
                image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
                cv2.drawFrameAxes(image, self.camera_params[camera]['kk'], self.camera_params[camera]['kc'], rvec, tvec, self.markers_size/2)
                rvec_refined, tvec_refined = cv2.solvePnPRefineLM(objectPoints=marker_points,
                                                      imagePoints=corner,
                                                      cameraMatrix=self.camera_params[camera]['kk'],
                                                      distCoeffs=self.camera_params[camera]['kc'],
                                                      rvec=rvec,
                                                      tvec=tvec,
                                                      criteria=(
                                                      cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-6))
                rot_matrix, _ = cv2.Rodrigues(rvec_refined)
                translation_matrix = np.eye(4)
                translation_matrix[:3, :3] = rot_matrix
                translation_matrix[:3, 3] = tvec_refined.flatten()

            if 'translation_matrix' not in self.data[camera][marker_id]:
                self.data[camera][marker_id]['translation_matrix'] = []

            self.data[camera][marker_id]['translation_matrix'].append(translation_matrix)

        if show:
            cv2.namedWindow('Detected markers {}'.format(camera), cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detected markers {}'.format(camera), 800, 600)
            cv2.imshow('Detected markers {}'.format(camera), image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def calculate_relative_pose(self, T_0_2, T_1_2):
        """
        Calculate the relative pose between two cameras given their transformation matrices,
        separating rotation and translation calculations.
         """
        # print(f"T_0_2:\n{T_0_2}")
        # print(f"T_1_2:\n{T_1_2}")

        # Invert the transformation matrix of the left camera
        T_2_1 = np.linalg.inv(T_1_2)
        # print(f"T_2_1:\n{T_2_1}")

        # Compute the relative transformation (right to left)
        T_1_0 = T_0_2 @ T_2_1
        # print(f"T_1_0:\n{T_1_0}")

        # Extract rotation and translation components
        R_0_2 = T_0_2[:3, :3]
        t_0_2 = T_0_2[:3, 3]
        R_1_2 = T_1_2[:3, :3]
        t_1_2 = T_1_2[:3, 3]

        # Compute relative rotation
        R_relative = R_0_2 @ R_1_2.T
        # print(f"R_relative:\n{R_relative}")

        # Compute relative translation
        t_relative = t_0_2 - R_relative @ t_1_2
        print(f"t_relative:\n{t_relative}")

        return R_relative, t_relative, T_1_0


    def calc_relative_pose_from_markers(self, Tcm1_list, Tcm2_list):
        """
        Calculate the relative pose between two markers given lists of transformation matrices of the camera and the markers.
        """
        if len(Tcm1_list) != len(Tcm2_list):
            raise ValueError("Tcm1_list and Tcm2_list must have the same length.")

        Tm1m2_list = []
        for Tcm1, Tcm2 in zip(Tcm1_list, Tcm2_list):
            Tcm1_inv = np.linalg.inv(Tcm1)
            Tm1m2 = Tcm1_inv @ Tcm2
            Tm1m2_list.append(Tm1m2)

        return Tm1m2_list

    def plot_camera_axes(self, R, t, ax, label, is_3d=True, projection_plane="XY"):
        """
        Plota os eixos da câmera nos gráficos 3D e 2D.
        """
        if label.split(' ')[0] == 'Left':
            color = 'purple'
            origin = np.zeros((3, 1))
        else:
            color = 'k'
            origin = t.flatten()

        # Vetores dos eixos
        x_axis = (origin + R[:, 0] * 100).flatten()
        y_axis = (origin + R[:, 1] * 100).flatten()
        z_axis = (origin + R[:, 2] * 100).flatten()

        if is_3d:
            ax.quiver(*origin, *(x_axis - origin), color='r')  # , label=f'{label} X-axis')
            ax.quiver(*origin, *(y_axis - origin), color='g')  # , label=f'{label} Y-axis')
            ax.quiver(*origin, *(z_axis - origin), color='b')  # , label=f'{label} Z-axis')
            ax.scatter(*origin, color=color, s=20, label=label)
        else:
            if projection_plane == "XY":
                ax.quiver(origin[0], origin[1], x_axis[0] - origin[0], x_axis[1] - origin[1], color='r')
                ax.quiver(origin[0], origin[1], y_axis[0] - origin[0], y_axis[1] - origin[1], color='g')
                ax.scatter(origin[0], origin[1], color=color, s=20, label=label)
            elif projection_plane == "XZ":
                ax.quiver(origin[0], origin[2], x_axis[0] - origin[0], x_axis[2] - origin[2], color='r')
                ax.quiver(origin[0], origin[2], z_axis[0] - origin[0], z_axis[2] - origin[2], color='b')
                ax.scatter(origin[0], origin[2], color=color, s=20, label=label)
            elif projection_plane == "YZ":
                ax.quiver(origin[0], origin[1], y_axis[0] - origin[1], y_axis[2] - origin[2], color='g')
                ax.quiver(origin[0], origin[2], z_axis[0] - origin[0], z_axis[2] - origin[2], color='b')
        ax.grid(True)


    def visualize_relative_pose(self, R_relative, t_relative, title='Camera pose'):
        fig = plt.figure(figsize=(12, 6))

        # Criar subplots: (1, 3) para layout horizontal
        ax1 = fig.add_subplot(131, projection='3d')  # Gráfico 3D
        ax2 = fig.add_subplot(132)  # Projeção XY
        ax3 = fig.add_subplot(133)  # Projeção XZ

        midpoint_rotation = cv2.Rodrigues(cv2.Rodrigues(R_relative)[0] / 2)[0]

        # Origem da câmera esquerda
        self.plot_camera_axes(midpoint_rotation.T, t_relative, ax1, label='Left Camera', is_3d=True)
        self.plot_camera_axes(midpoint_rotation, t_relative, ax1, label='Right Camera', is_3d=True)

        # Projeção no plano XY
        self.plot_camera_axes(midpoint_rotation.T, t_relative, ax2, label='Left Camera', is_3d=False)
        self.plot_camera_axes(midpoint_rotation, t_relative, ax2, label='Right Camera', is_3d=False)

        # Projeção no plano XZ
        self.plot_camera_axes(midpoint_rotation.T, t_relative, ax3, label='Left Camera', is_3d=False,
                              projection_plane="XZ")
        self.plot_camera_axes(midpoint_rotation, t_relative, ax3, label='Right Camera', is_3d=False,
                              projection_plane="XZ")

        # Configurações do gráfico 3D
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Full 3D View \n {}'.format(title))

        # Configurações das projeções 2D
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Y [mm]')
        ax2.set_title('Top View (XY)')
        ax3.set_xlabel('X [mm]')
        ax3.set_ylabel('Z [mm]')
        ax3.set_title('Side View (XZ)')

        # **Cálculo automático dos limites compactos**
        def get_limits(val1, val2, margin=0.2):
            """ Define limites para um eixo com base nos dois pontos. """
            min_val = min(val1, val2)
            max_val = max(val1, val2)
            range_val = max_val - min_val
            padding = margin * range_val if range_val > 0 else 50  # Evita zero
            return (min_val - padding, max_val + padding)

        ax1.set_xlim(get_limits(0, t_relative[0]))
        ax1.set_ylim(get_limits(0, t_relative[0]))
        ax1.set_zlim(get_limits(0, t_relative[0]))
        ax1.legend()

        ax2.set_xlim(get_limits(0, t_relative[0]))
        ax2.set_ylim(get_limits(0, t_relative[1]))
        ax2.legend()

        ax3.set_xlim(get_limits(0, t_relative[0]))
        ax3.set_ylim(get_limits(0, t_relative[2]))
        ax3.legend()

        plt.tight_layout()
        plt.show()


    def transform_precision(self, transforms):

        if all(transforms is not None for transforms in transforms):
            positions = [transform[:3, 3] for transform in transforms]
            rotations = [cv2.Rodrigues(transform[:3, :3])[0] for transform in transforms]

            # Ensure positions is a NumPy array
            positions = np.array(positions)

            if positions.size == 0:
                raise ValueError("The positions array is empty.")

            pos_range = np.ptp(positions, axis=0)
            rot_range = np.ptp(rotations, axis=0)
            print(f"Position range: {np.mean(pos_range)}")
            print(f"Rotation range: {np.mean(rot_range)}")
            if np.mean(pos_range) > 0.6:
                print("Transformations are not precise enough.")
                return None
            else:
                translation_matrix = np.eye(4)
                translation_matrix[:3, :3] = cv2.Rodrigues(np.mean(rotations, axis=0))[0]
                translation_matrix[:3, 3] = np.mean(positions, axis=0).flatten()
                return translation_matrix
        else:
            return
