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
    yaml_file = './camera_parameters/calibration_params.yaml'
    extrinsic = ExtrinsicCalib(yaml_file, markers_size=95)
    extrinsic.read_yaml_file(yaml_file)

    # extrinsic.generat_aruco_markers(markers_path='./markers')
    
    # Caminho para a imagem que será usada para detectar os marcadores ArUco
    path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/Calib Extrinseca/20250319'
    left_images_names = sorted(os.listdir(os.path.join(path, 'left')))
    right_images_names = sorted(os.listdir(os.path.join(path, 'right')))
    left_iamges = [cv2.imread(os.path.join(path, 'left', image)) for image in left_images_names]
    right_images = [cv2.imread(os.path.join(path, 'right', image)) for image in right_images_names]

    t_left = []
    t_right = []
    for left_img, right_img in zip(left_iamges, right_images):

        # Detectar os marcadores ArUco na imagem
        corners_l, ids_l = extrinsic.detect_aruco_markers(left_img)
        corners_r, ids_r = extrinsic.detect_aruco_markers(right_img)

        if ids_l is None and ids_r is None:
            print("Nenhum marcador ArUco detectado.")
        else:
            print(f"Marcadores ArUco detectados: {ids_l.flatten()}")
            print(f"Marcadores ArUco detectados: {ids_r.flatten()}")
            extrinsic.show_stereo_detection_markers(left=left_img, right=right_img, corners_left=corners_l, corners_right=corners_r, ids_left=ids_l, ids_right=ids_r)



    # # Mostrar os marcadores detectados
        # extrinsic.show_detected_markers(image, corners, ids)
        extrinsic.solve_pnp_maker(ids_l, corners_l, camera='left')
        extrinsic.solve_pnp_maker(ids_r, corners_r, camera='right')
        data = extrinsic.data
    print(data)

if __name__ == '__main__':
    # charuco_calibration()
    aruco_markers()
    # extrinsic_calibration()