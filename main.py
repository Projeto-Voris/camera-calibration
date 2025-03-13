from include import charuco_calibration


def main():
    charuco = charuco_calibration.CharucoCalibration(squares=(12, 8), marker_size=(28, 18))  # Sent to print
    # charuco = charuco_calibration.CharucoCalibration(squares=(5, 10), dpi=600, paper_size=(297, 210), marker_size=(30, 20)) # images 21/10
    charuco.create_charucoboard(min_margin_mm=5, paper_size=(297, 420), dpi=600, save_as_pdf=False)

    # Save the board as an image for printing

    # Display the board image
    # charuco.view_charuco_board()

    # cv2.imwrite('charuco_a3_600dpi.png', board_img)

    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/Calib Extrinseca/20250311/S1'
    charuco.load_image(images_path=images_path, visualize=False)
    charuco.detect_charuco_board(camera='left', visualize=True)
    charuco.detect_charuco_board(camera='right', visualize=True)
    charuco.calibration_camera(camera='left')
    charuco.calibration_camera(camera='right')
    charuco.stereo_calibration()
    charuco.save_camera_params(path='./calibration_params.yaml')
    print('debug')
    # charuco.show_undistorted_image(camera='left')
    charuco.homogeneous_matrix(camera='left', visualize=False)
    # print(charuco.all_ids['right'])
    print('debugger')


if __name__ == '__main__':
    main()
