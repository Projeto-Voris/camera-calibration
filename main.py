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
    extrinsic = ExtrinsicCalib(yaml_file)
    extrinsic.read_yaml_file(yaml_file)
    
    # Caminho para a imagem que será usada para detectar os marcadores ArUco
    image_path = '/home/tiago/Downloads/Calib Extrinseca-20250312T134809Z-001/Calib Extrinseca/20250311/S1/left/L000.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return
    
    # Detectar os marcadores ArUco na imagem
    image, corners, ids = extrinsic.detect_aruco_markers(image)
    
    if ids is None:
        print("Nenhum marcador ArUco detectado.")
    else:
        print(f"Marcadores ArUco detectados: {ids.flatten()}")
    
    # # Mostrar os marcadores detectados
    # extrinsic.show_detected_markers(image, corners, ids)

    extrinsic.solve_pnp_maker(ids, corners, camera='left')

if __name__ == '__main__':
    # charuco_calibration()
    aruco_markers()
    # extrinsic_calibration()