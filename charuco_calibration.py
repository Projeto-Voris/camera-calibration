import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class CharucoCalibration:
    def __init__(self, squares=(3, 4), paper_size=(210, 297), dpi=300, marker_size=(60, 40),
                 aruco_d=cv2.aruco.DICT_5X5_250):
        """
        Class to calibrate stereo cameras using ChArUco calibration board
        Parameters:
            squares: tuple(int, int) Number of squares in board (row, col)
            paper_size: tuple(int, int) Size of paper used (width, height) [mm]
            dpi: int Dots per inch for printer specs.
            marker_size: tuple(int, int) Size of square and ArUco marker (square, marker) [px]
            aruco_d: ArUco dictionary

        """
        self.squares = squares
        self.paper_size = paper_size
        self.marker_size = marker_size
        self.dpi = dpi
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_d)
        self.charuco_board = None
        self.board_image = self.create_charucoboard()

        self.images_left = np.array([])
        self.images_right = np.array([])
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([]),
                     'std_kk': np.array([]), 'std_kc': np.array([])},

            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([]),
                      'std_kk': np.array([]), 'std_kc': np.array([])},

            'stereo': {'R': np.array([]), 'T': np.array([])}
        }

        self.references_ids = None  # variable to store IDs from first image.
        self.all_corners = {'left': [], 'right': []}
        self.all_ids = {'left': [], 'right': []}

    def create_charucoboard(self, margin_mm=10, save_as_pdf=True):
        """
        Function to create ChArUco board with margins and save it as PDF.

        Args:
            margin_mm: The margin size around the ChArUco board in millimeters.
            save_as_pdf: Boolean, whether to save the image as a PDF.

        Returns:
            ChArUco board image with margins (height, width).
        """
        # Convert the square and marker size from millimeters to pixels
        square_length_px = int(np.ceil(self.marker_size[0] / 25.4 * self.dpi))
        marker_length_px = int(np.ceil(self.marker_size[1] / 25.4 * self.dpi))

        # Create the ChArUco board with pixel-based dimensions
        self.charuco_board = cv2.aruco.CharucoBoard(
            size=self.squares,
            squareLength=square_length_px,
            markerLength=marker_length_px,
            dictionary=self.aruco_dict
        )

        # Convert paper dimensions from millimeters to pixels
        paper_px = [int(np.ceil(a / 25.4 * self.dpi)) for a in self.paper_size]

        # Generate the ChArUco board image
        board_image = self.charuco_board.generateImage(paper_px)

        # Convert the margin size from millimeters to pixels
        margin_px = int(np.ceil(margin_mm / 25.4 * self.dpi))

        # Create a larger canvas with margins (white background)
        board_with_margin = np.full(
            (board_image.shape[0] + 2 * margin_px, board_image.shape[1] + 2 * margin_px),
            255, dtype=np.uint8  # White background
        )

        # Place the ChArUco board image in the center of the canvas
        board_with_margin[margin_px:-margin_px, margin_px:-margin_px] = board_image

        if save_as_pdf:
            # Set the figure size to match the paper size exactly
            fig = plt.figure(figsize=(self.paper_size[0] / 25.4, self.paper_size[1] / 25.4), dpi=self.dpi)
            plt.imshow(board_with_margin, cmap="gray", interpolation="nearest")
            plt.axis('off')

            # Save the figure as a PDF without cropping or additional padding
            plt.savefig("charuco_board.pdf")
            plt.close(fig)

        return board_with_margin


    def get_charuco_image(self):
        """
        Function to get ChArUco board image to print
        Returns:
            (height, width) ChArUco board image to print
        """
        return self.board_image

    def load_image(self, images_path, visualize=False):
        """
        Function to load image from given path.
        Parameters:
        image_path: path to image
        visualize: boolean, if True image will be displayed
        Returns:
            Save images inside class variable
        """
        if os.path.exists(images_path):
            images_left = sorted(os.listdir(os.path.join(images_path, 'left')))
            images_right = sorted(os.listdir(os.path.join(images_path, 'right')))
            if len(images_left) != len(images_right):
                raise Exception('Number of images do not match')

            print('Found {} images on provided path'.format(len(images_left) + len(images_right)))

            # Read images and create a list of it.
            left_img = [cv2.imread(os.path.join(images_path, 'left', str(img_name)), cv2.IMREAD_GRAYSCALE)
                        for img_name in images_left]
            right_img = [cv2.imread(os.path.join(images_path, 'right', str(img_name)), cv2.IMREAD_GRAYSCALE)
                         for img_name in images_right]

            # Convert to numpy array
            self.images_left = np.stack(left_img, axis=-1).astype(np.uint8)  # Convert to uint8
            self.images_right = np.stack(right_img, axis=-1).astype(np.uint8)  # Convert to uint8

        if visualize:
            for k in range(len(self.images_left)):
                concat = np.concatenate((self.images_left[k], self.images_right[k]), axis=1)
                cv2.namedWindow('Calibration images', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Calibration images', concat.shape[1] // 2, concat.shape[0] // 2)
                cv2.imshow('Calibration images', concat)
                cv2.waitKey(0)
            cv2.destroyWindow('Calibration images')

    def detect_charuco_board(self, camera, visualize=False):
        if camera == 'left':
            images = self.images_left
        elif camera == 'right':
            images = self.images_right
        else:
            raise Exception('Camera must be either left or right')
        for i in range(images.shape[2]):
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(image=images[:, :, i], dictionary=self.aruco_dict)
            bgr_img = cv2.cvtColor(images[:, :, i], cv2.COLOR_GRAY2BGR)
            if ids is None:
                print('No detected markers on image {} from {} camera'.format(i, camera))
                continue
            elif len(ids) > 0:
                response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                                             markerIds=ids,
                                                                                             image=images[:, :, i],
                                                                                             board=self.charuco_board)
                # If true will show detected markers and corners images
                if visualize:
                    img_marker = cv2.aruco.drawDetectedMarkers(image=bgr_img, corners=corners, ids=ids)
                    img_corners = cv2.aruco.drawDetectedCornersCharuco(image=img_marker.copy(), charucoCorners=charuco_corners,
                                                                       charucoIds=charuco_ids)
                    concat = np.hstack((img_marker, img_corners))
                    cv2.namedWindow('Detected images', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Detected images', concat.shape[1] // 2, concat.shape[0] // 2)
                    cv2.imshow('Detected images', concat)
                    cv2.waitKey(0)
                    cv2.destroyWindow('Detected images')
                # Save charuco corners id for first image as reference
                if i == 0:
                    self.references_ids = set(charuco_ids.flatten())
                if response and charuco_corners is not None:
                    filtered_corners = []
                    filtered_ids = []
                    for corners, corner_id in zip(charuco_corners, charuco_ids):
                        if corner_id[0] in self.references_ids:
                            filtered_corners.append(corners)
                            filtered_ids.append(corner_id)
                    if len(filtered_corners) > 2:
                        self.all_corners[camera].append(np.array(filtered_corners, dtype=np.float32))
                        self.all_ids[camera].append(np.array(filtered_ids, dtype=np.int32))

    def calibration_camera(self, camera, visualize=False):
        if len(self.all_corners[camera]) > 0:
            ret, kk, kc, rvecs, tvecs, std_kk, std_kc = cv2.aruco.calibrateCameraCharucoExtended(
                charucoCorners=self.all_corners[camera],
                charucoIds=self.all_ids[camera],
                board=self.charuco_board,
                imageSize=(self.images_left.shape[1], self.images_left.shape[0]))
            self.camera_params[camera]['kk'] = kk
            self.camera_params[camera]['kc'] = kc
            self.camera_params[camera]['rvecs'] = rvecs[0]
            self.camera_params[camera]['tvecs'] = tvecs[0]
            self.camera_params[camera]['std_kk'] = std_kk
            self.camera_params[camera]['std_kc'] = std_kc
