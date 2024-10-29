import cv2
import charuco_calibration


def main():
    charuco = charuco_calibration.CharucoCalibration(squares=(8, 6), marker_size=(30, 20)) #Sent to print
    # charuco = charuco_calibration.CharucoCalibration(squares=(5, 10), dpi=600, paper_size=(297, 210), marker_size=(30, 20)) # images 21/10
    board_img = charuco.create_charucoboard(min_margin_mm=5, paper_size=(210, 297), dpi=300, save_as_pdf=False)

    # Save the board as an image for printing

    # Display the board image
    cv2.namedWindow('Charuco', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Charuco', board_img.shape[1]//4, board_img.shape[0]//4)
    cv2.imshow('Charuco', board_img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('charuco_a3_600dpi.png', board_img)

    # images_path = '/home/daniel/PycharmProjects/camera_calibration/images/20241021'
    # charuco.load_image(images_path=images_path, visualize=False)
    # charuco.detect_charuco_board(camera='left', visualize=True)
    # charuco.detect_charuco_board(camera='right', visualize=False)
    # charuco.calibration_camera(camera='left', visualize=False)
    # charuco.calibration_camera(camera='right', visualize=False)

    # print(charuco.all_ids['right'])


if __name__ == '__main__':
    main()
