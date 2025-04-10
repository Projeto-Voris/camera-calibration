import os.path
import numpy as np
import cv2
from src.charuco_calibration import CharucoCalibration
from src.extrinsic_calibration import ExtrinsicCalib

def load_images(path, cameras=('left', 'right')):
    images = {}
    for camera in cameras:
        camera_path = os.path.join(path, camera)
        images[camera] = [cv2.imread(os.path.join(camera_path, img)) for img in sorted(os.listdir(camera_path))]
    return images

def charuco_calibration():
    charuco = CharucoCalibration(squares=(12, 8), marker_size=(28, 18))  # Sent to print
    # charuco = charuco_calibration.CharucoCalibration(squares=(5, 10), dpi=600, paper_size=(297, 210), marker_size=(30, 20)) # images 21/10
    charuco.create_charucoboard(min_margin_mm=5, paper_size=(297, 420), dpi=600, save_as_pdf=False)

    # Save the board as an image for printing

    # Display the board image
    # charuco.view_charuco_board()

    # cv2.imwrite('charuco_a3_600dpi.png', board_img)

    images_path = '/home/tiago/Downloads/Calib Extrinseca-20250312T134809Z-001/Calib Extrinseca/20250311/S2'
    charuco.load_image(images_path=images_path, visualize=False)
    charuco.detect_charuco_board(camera='left', visualize=True)
    charuco.detect_charuco_board(camera='right', visualize=True)
    charuco.calibration_camera(camera='left')
    charuco.calibration_camera(camera='right')
    charuco.stereo_calibration()
    charuco.save_camera_params(path='./camera_parameters/sm4_matlab.yaml')
    print('debug')
    # charuco.show_undistorted_image(camera='left')
    charuco.homogeneous_matrix(camera='left', visualize=False)
    # print(charuco.all_ids['right'])
    print('debugger')

def aruco_markers():
    yaml_file = './camera_parameters/sm3_matlab.yaml'
    extrinsic = ExtrinsicCalib(yaml_file, markers_size=95)
    extrinsic.read_yaml_file(yaml_file)

    # Caminho para as imagens
    path = '/home/tiago/Pictures/image_logs/SM3/20250404_1_2_3'
    
    images = load_images(path)
    left_images = images['left']
    right_images = images['right']

    # Detectar os marcadores ArUco nas imagens
    for i, (left, right) in enumerate(zip(left_images, right_images)):
        corners_l, ids_l = extrinsic.detect_aruco_markers(left)
        corners_r, ids_r = extrinsic.detect_aruco_markers(right)

        # print("Detected marker id {} img L {}:".format(ids_l.flatten(), i))
        # print("Detected marker id {} img R {}:".format(ids_r.flatten(), i))

        # extrinsic.show_detected_markers(left_images[0], corners_l, ids_l)
        # extrinsic.show_detected_markers(right_images[0], corners_r, ids_r)

        # Estimar a pose para os marcadores detectados
        extrinsic.pose_estimation(ids_l.flatten(), corners_l, camera='left', image=left, show=True)
        extrinsic.pose_estimation(ids_r.flatten(), corners_r, camera='right', image=right, show=True)


    # Acessar os dados armazenados
    data = extrinsic.data

    T_relative_list = []

    valid_markers = extrinsic.get_valid_markers()
    # Calcular a pose relativa entre as câmeras usando o primeiro marcador detectado
    for marker_id in valid_markers:
        print(f"Marker ID: {marker_id}")

        T_l_m = extrinsic.transform_precision(data['left'][marker_id]['translation_matrix'])
        T_r_m = extrinsic.transform_precision(data['right'][marker_id]['translation_matrix'])

        # Calcular a pose relativa entre as câmeras para o marcador atual
        if T_l_m is not None and T_r_m is not None:
            R_relative, t_relative, T_relative = extrinsic.calculate_relative_pose(T_l_m, T_r_m)
            if abs(t_relative[2]) > 55 and abs(t_relative[1]) > 55:
                print('Invalid translation for marker {}: {}'.format(marker_id, t_relative))
                continue
            T_relative_list.append(T_relative)

            # Visualizar a pose relativa para o marcador atual
            extrinsic.visualize_relative_pose(R_relative, t_relative, title="Marker ID: {} path: {}".format(marker_id, path.split('/')[-1]))
            
def multi_aruco_markers():

    # Caminho para as imagens
    path_full = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/Calib Extrinseca/20250404_multi'
    for folder in os.listdir(path_full):
        path = os.path.join(path_full, folder)
        if folder == "SM3":
                print('========= SM3 =========')
                yaml_file = './camera_parameters/sm3_matlab.yaml'
                extrinsic = ExtrinsicCalib(yaml_file, markers_size=95, aruco_d=cv2.aruco.DICT_5X5_100)
                extrinsic.read_yaml_file(yaml_file)
                T_SM3 = np.zeros((4, 4))
                T_SM3[:3,:3] = extrinsic.camera_params['stereo']['R']
                T_SM3[:3,3] = extrinsic.camera_params['stereo']['T']

        elif folder == "SM2":
            print('========= SM2 =========')
            yaml_file = 'camera_parameters/sm2_matlab.yaml'
            extrinsic = ExtrinsicCalib(yaml_file, markers_size=95, aruco_d=cv2.aruco.DICT_5X5_100)
            extrinsic.read_yaml_file(yaml_file)
            T_SM2 = np.zeros((4, 4))
            T_SM2[:3,:3] = extrinsic.camera_params['stereo']['R']
            T_SM2[:3,3] = extrinsic.camera_params['stereo']['T']

        images = load_images(path)
        left_images = images['left']
        right_images = images['right']

        # Detectar os marcadores ArUco nas imagens
        for i, (left, right) in enumerate(zip(left_images, right_images)):
            corners_l, ids_l = extrinsic.detect_aruco_markers(left)
            corners_r, ids_r = extrinsic.detect_aruco_markers(right)

            if ids_l is None or ids_r is None:
                print('No markers detected in image {}.'.format(i))
                continue
            # print("Detected marker id {} img L {}:".format(ids_l.flatten(), i))
            # print("Detected marker id {} img R {}:".format(ids_r.flatten(), i))

            # extrinsic.show_detected_markers(left_images[0], corners_l, ids_l)
            # extrinsic.show_detected_markers(right_images[0], corners_r, ids_r)

            # Estimar a pose para os marcadores detectados
            extrinsic.pose_estimation(ids_l.flatten(), corners_l, camera='left', image=left, show=False)
            extrinsic.pose_estimation(ids_r.flatten(), corners_r, camera='right', image=right, show=False)


        # Acessar os dados armazenados
        data = extrinsic.data

        T_relative_list_01 = []
        T_relative_list_10 = []

        valid_markers = extrinsic.get_valid_markers()
        # Calcular a pose relativa entre as câmeras usando o primeiro marcador detectado
        for marker_id in valid_markers:
            print(f"--- Marker ID: {marker_id}")

            T_l_m = extrinsic.transform_precision(data['left'][marker_id]['translation_matrix'])
            T_r_m = extrinsic.transform_precision(data['right'][marker_id]['translation_matrix'])

            # Calcular a pose relativa entre as câmeras para o marcador atual
            if T_l_m is not None and T_r_m is not None:
                R_relative_01, t_relative_01, T_relative_01 = extrinsic.calculate_relative_pose_0to1(T_l_m, T_r_m)
                R_relative_10, t_relative_10, T_relative_10 = extrinsic.calculate_relative_pose_1to0(T_l_m, T_r_m)
                # if abs(t_relative[2]) > 55 and abs(t_relative[1]) > 55:
                #     print('Invalid translation for marker {}: {}'.format(marker_id, t_relative))
                #     continue
                T_relative_list_01.append(T_relative_01)
                T_relative_list_10.append(T_relative_10)
 
                # Visualizar a pose relativa para o marcador atual
                # extrinsic.visualize_relative_pose(R_relative_01, t_relative_01, title="T01 Marker ID: {} path: {}".format(marker_id, path.split('/')[-1]))
                # extrinsic.visualize_relative_pose(R_relative_10, t_relative_10, title="T10 Marker ID: {} path: {}".format(marker_id, path.split('/')[-1]))

            # print('precision relative transform:')
            # extrinsic.transform_precision(T_relative_list)
            # print('debug:')
            # print(T_relative_list)


        # # Calcula precisão entre marcadores

        # T_mean = np.mean(T_relative_list, axis=0)
        # # print('precision between cameras: \n{}'.format(T_mean))
        # # Calcular a pose relativa entre os marcadores
        # extrinsic.visualize_relative_pose(T_mean[:3,:3], T_mean[:3,3].flatten(), title="Mean Relative Pose")

def charuco_markers():
    yaml_file = './camera_parameters/sm3_charuco.yaml'
    extrinsic = ExtrinsicCalib(yaml_file, markers_size=(28, 18))
    extrinsic.read_yaml_file(yaml_file)

    # Caminho para as imagens
    path = '/home/tiago/Downloads/Calib Extrinseca-20250312T134809Z-001/Calib Extrinseca/20250311/S1'
    
    images = load_images(path)
    left_images = images['left']
    right_images = images['right']

    # Detectar os marcadores ArUco nas imagens
    for i, (left, right) in enumerate(zip(left_images, right_images)):
        T_relative = []
        corners_l, ids_l = extrinsic.detect_charuco_board(left)
        corners_r, ids_r = extrinsic.detect_charuco_board(right)
        obj_l, img_l = extrinsic.match_charuco_points(corners_l, ids_l)
        obj_r, img_r = extrinsic.match_charuco_points(corners_r, ids_r)
        rvec_l, tvec_l, T_L = extrinsic.pose_estimation_charuco(camera='left', obj_points=obj_l, img_points=img_l)
        rvec_r, tvec_r, T_R = extrinsic.pose_estimation_charuco(camera='right', obj_points=obj_r, img_points=img_r)

        # Calculate relative pose
        _,_, T_relative = extrinsic.calculate_relative_pose(T_L, T_R)
        print("====== Img: {} ======".format(i))
        print(T_relative[:3, 3])
        extrinsic.visualize_relative_pose(T_relative[:3, :3], T_relative[:3, 3], title="Image: {} path: {}".format(i, path.split('/')[-1]))

if __name__ == '__main__':
    # charuco_calibration()
    # aruco_markers()
    # multi_aruco_markers()
    charuco_markers()
