import cv2
import charuco_calibration


def main():
    charuco = charuco_calibration.CharucoCalibration(squares=(7, 8), paper_size=(297, 420), marker_size=(60, 40))

    board_img = charuco.get_charuco_image()

    # Save the board as an image for printing

    # Display the board image
    cv2.namedWindow('Charuco', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Charuco', 800, 600)
    cv2.imshow('Charuco', board_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('charuco_a4_300dpi.png', board_img)

    # images_path = '/home/daniel/PycharmProjects/camera_calibration/images/20241021'
    # charuco.load_image(images_path=images_path, visualize=False)
    # charuco.detect_charuco_board(camera='left', visualize=True)
    # charuco.detect_charuco_board(camera='right', visualize=False)
    # charuco.calibration_camera(camera='left', visualize=False)
    # charuco.calibration_camera(camera='right', visualize=False)

    # print(charuco.all_ids['right'])


if __name__ == '__main__':
    main()
