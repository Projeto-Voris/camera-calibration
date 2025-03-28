import os.path

import cv2
from src.charuco_calibration import CharucoCalibration
from src.extrinsic_calibration import ExtrinsicCalib

def charuco_calibration():
    charuco = CharucoCalibration(squares=(12, 8), marker_size=(28, 18))  # Sent to print
    # charuco = charuco_calibration.CharucoCalibration(squares=(5, 10), dpi=600, paper_size=(297, 210), marker_size=(30, 20)) # images 21/10
    charuco.create_charucoboard(min_margin_mm=5, paper_size=(297, 420), dpi=600, save_as_pdf=False)

    # Save the board as an image for printing

    # Display the board image
    # charuco.view_charuco_board()

    # cv2.imwrite('charuco_a3_600dpi.png', board_img)

    images_path = '/home/tiago/Downloads/Calib Extrinseca-20250312T134809Z-001/Calib Extrinseca/20250311/S1'
    charuco.load_image(images_path=images_path, visualize=False)
    charuco.detect_charuco_board(camera='left', visualize=True)
    charuco.detect_charuco_board(camera='right', visualize=True)
    charuco.calibration_camera(camera='left')
    charuco.calibration_camera(camera='right')
    charuco.stereo_calibration()
    charuco.save_camera_params(path='./camera_parameters/calibration_params.yaml')
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
    path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/Calib Extrinseca/20250320'
    left_images_names = sorted(os.listdir(os.path.join(path, 'left')))
    right_images_names = sorted(os.listdir(os.path.join(path, 'right')))
    left_images = [cv2.imread(os.path.join(path, 'left', image)) for image in left_images_names]
    right_images = [cv2.imread(os.path.join(path, 'right', image)) for image in right_images_names]

    
    # Detectar os marcadores ArUco nas imagens
    for i, (left, right) in enumerate(zip(left_images, right_images)):
        corners_l, ids_l = extrinsic.detect_aruco_markers(left)
        corners_r, ids_r = extrinsic.detect_aruco_markers(right)

        print("Detected marker id {} img L {}:".format(ids_l.flatten(), i))
        print("Detected marker id {} img R {}:".format(ids_r.flatten(), i))

    # extrinsic.show_detected_markers(left_images[0], corners_l, ids_l)
    # extrinsic.show_detected_markers(right_images[0], corners_r, ids_r)

    # Estimar a pose para os marcadores detectados
        extrinsic.pose_estimation(ids_l.flatten(), corners_l, camera='left', image=left, show=True)
        extrinsic.pose_estimation(ids_r.flatten(), corners_r, camera='right', image=right, show=True)


    # Acessar os dados armazenados
    data = extrinsic.data

    tm0l = extrinsic.transform_precision(data['left'][1]['translation_matrix'])
    tm0r = extrinsic.transform_precision(data['right'][1]['translation_matrix'])

    # Calcular a pose relativa entre as câmeras usando o primeiro marcador detectado
    for marker_id in data['left'].keys():
        if marker_id in data['right']:
            translation_matrix_left = data['left'][marker_id]['translation_matrix'][-1]
            translation_matrix_right = data['right'][marker_id]['translation_matrix'][-1]

            # Calcular a pose relativa entre as câmeras para o marcador atual
            R_relative, t_relative = extrinsic.calculate_relative_pose(translation_matrix_left, translation_matrix_right)

            # Visualizar a pose relativa para o marcador atual
            print(f"Marker ID: {marker_id}")
            extrinsic.transform_precision()

if __name__ == '__main__':
    # charuco_calibration()
    aruco_markers()
    # extrinsic_calibration()