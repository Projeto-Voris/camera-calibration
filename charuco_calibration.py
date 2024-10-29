import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class CharucoCalibration:
    def __init__(self, squares=(8,6), marker_size=(60, 40),
                 aruco_d=cv2.aruco.DICT_6X6_250):
        """
        Class to calibrate stereo cameras using ChArUco calibration board
        Parameters:
            squares: Number of squares in board (row, col)
            paper_size: Size of paper used (width, height) [mm]
            dpi: Dots per inch for printer specs.
            marker_size: Size of ChArUcos (square, marker) [mm]
            aruco_d: ArUco dictionary

        """
        self.squares = squares
        self.marker_size = marker_size
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_d)

        # Create charuco board
        self.charuco_board = cv2.aruco.CharucoBoard(
            size=self.squares,
            squareLength=marker_size[0],  # mm
            markerLength=marker_size[1],  # mm
            dictionary=self.aruco_dict
        )
        print("ChArUco board size:: {}".format(self.squares))
        print('Dimension of board: {} mm'.format(self.marker_size[0]))
        print('Dimension of marker: {} mm'.format(self.marker_size[1]))
        print('Dict: {`}', format(str(self.aruco_dict)))

        # Create board to output
        self.board_image = None

        # Images variables
        self.images_left = np.array([])
        self.images_right = np.array([])

        # Charuco image's parameters
        self.references_ids = None  # variable to store IDs from first image.
        self.all_corners = {'left': [], 'right': []}
        self.all_ids = {'left': [], 'right': []}

        # Calibration results
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([]),
                     'std_kk': np.array([]), 'std_kc': np.array([])},

            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([]),
                      'std_kk': np.array([]), 'std_kc': np.array([])},

            'stereo': {'R': np.array([]), 'T': np.array([])}
        }

    def create_charucoboard(self, min_margin_mm=10, paper_size=(210, 297), dpi=300,
                            save_as_pdf=True):
        """
        Create ChArUco board
        Parameters:
            :param min_margin_mm: minimum margin in mm
            :param paper_size: paper size (width, height)
            :param dpi: dots per inch for printer specs.
            :param orientation: 'landscape' or 'portrait' for board orientation
            :param save_as_pdf: boolean to save image as pdf or not
        Returns:
           board: image of charuco board
        """
        print('{} squares: {} mm'.format(self.squares, tuple(sq*self.marker_size[0] for sq in self.squares)))
        print('Paper size: {}'.format(paper_size))

        if (paper_size[0] <= np.ceil(self.squares[0] * self.marker_size[0] + 2 * min_margin_mm)
                or paper_size[1] <= np.ceil(self.squares[1] * self.marker_size[0] + 2 * min_margin_mm)):
            print('Portrait dont fit, change to landscape')
            paper_size = (paper_size[1], paper_size[0])
            print('Paper size: {}'.format(paper_size))

            if (paper_size[0] <= np.ceil(self.squares[0] * self.marker_size[0] + 2 * min_margin_mm)
                    or paper_size[1] <= np.ceil(self.squares[1] * self.marker_size[0] + 2 * min_margin_mm)):
                raise ValueError('Paper size too small')
            else:
                print('Landscape fit')
        else:
            print('Portrait fit')


        # Calculate square and marker sizes in pixels
        square_length_px = int(np.round(self.marker_size[0] / 25.4 * dpi))
        marker_length_px = int(np.round(self.marker_size[1] / 25.4 * dpi))

        # Create Charuco board
        charuco_board = cv2.aruco.CharucoBoard(
            size=self.squares,
            squareLength=square_length_px,
            markerLength=marker_length_px,
            dictionary=self.aruco_dict
        )

        # Calculate total board size in pixels
        board_width_px = square_length_px * self.squares[0]
        board_height_px = square_length_px * self.squares[1]

        # Convert minimum margin to pixels
        min_margin_px = int(np.round(min_margin_mm / 25.4 * dpi))

        # Convert paper dimensions to pixels
        paper_width_px = int(np.round(paper_size[0] / 25.4 * dpi))
        paper_height_px = int(np.round(paper_size[1] / 25.4 * dpi))

        # Ensure board fits on paper by adjusting margins if necessary
        margin_x = max(min_margin_px, (paper_width_px - board_width_px) // 2)
        margin_y = max(min_margin_px, (paper_height_px - board_height_px) // 2)

        # Adjust paper size if the board plus margins exceed it
        if board_width_px + 2 * margin_x > paper_width_px or board_height_px + 2 * margin_y > paper_height_px:
            raise ValueError('Something went wrong with margin')


        # Create a blank canvas with the adjusted paper size
        canvas = np.full((paper_height_px, paper_width_px), 255, dtype=np.uint8)

        # Generate the board image and place it on the canvas
        board_image = charuco_board.generateImage((board_width_px, board_height_px))
        canvas[margin_y:margin_y + board_image.shape[0], margin_x:margin_x + board_image.shape[1]] = board_image

        # Save to PDF if specified
        if save_as_pdf:
            # Convert paper size to inches for Matplotlib figsize
            fig_width_in = round(paper_size[0] / 25.4,1)
            fig_height_in = round(paper_size[1] / 25.4,1)

            # Set up the figure size and save the PDF
            fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
            plt.imshow(canvas, cmap='gray', interpolation='nearest')
            plt.axis('off')

            # Save the PDF without extra padding or scaling
            plt.savefig("charuco_board.pdf", format='pdf', dpi=dpi)
            plt.close(fig)

        cv2.imwrite('charuco_{}x{}_size_{}_{}.png'.format(self.squares[0], self.squares[1],
                                                          self.marker_size[0], self.marker_size[1]), canvas)

        print('Resolution: {} pixel/mm'.format(round(canvas.shape[0] / paper_size[1], 2)))

        return canvas

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
                continue
            elif len(ids) > 0:
                print('No detected markers on image {} from {} camera'.format(i, camera))
                response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                                             markerIds=ids,
                                                                                             image=images[:, :, i],
                                                                                             board=self.charuco_board)
                # If true will show detected markers and corners images
                if visualize:
                    img_marker = cv2.aruco.drawDetectedMarkers(image=bgr_img, corners=corners, ids=ids)
                    img_corners = cv2.aruco.drawDetectedCornersCharuco(image=img_marker.copy(),
                                                                       charucoCorners=charuco_corners,
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
