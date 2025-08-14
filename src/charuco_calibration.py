import os
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt


class CharucoCalibration:
    def __init__(self, squares: tuple[int, int] = (8, 6), marker_size: tuple[int, int] = (60, 40),
                 aruco_d: int = cv2.aruco.DICT_4X4_100) -> None:
        """
        Class to calibrate stereo cameras using ChArUco calibration board
            :param squares: Number of squares in board (row, col)
            :param paper_size: Size of paper used (width, height) [mm]
            :param dpi: Dots per inch for printer specs.
            :param marker_size: Size of ChArUcos (square, marker) [mm]
            :param aruco_d: ArUco dictionary

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
        self.object_points = []
        print("ChArUco board size:: {}".format(self.squares))
        print('Dimension of board: {} mm'.format(self.marker_size[0]))
        print('Dimension of marker: {} mm'.format(self.marker_size[1]))
        print('Dict: {`}', format(str(self.aruco_dict)))

        # Create board to output
        self.board_image = None

        # Images variables
        self.images_path = None
        self.images_left = np.array([])
        self.images_right = np.array([])

        # Charuco image's parameters
        self.references_ids = None  # variable to store IDs from first image.
        self.all_corners = {'left': [], 'right': []}
        self.all_ids = {'left': [], 'right': []}

        # Calibration results
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([]), 'H': np.array([])},
            # 'std_kk': np.array([]), 'std_kc': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([]), 'H': np.array([])},
            # 'std_kk': np.array([]), 'std_kc': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([]), 'E': np.array([]), 'F': np.array([])}
        }

    def create_charucoboard(self, min_margin_mm: int = 10, paper_size: tuple[int, int] = (210, 297), dpi: int = 300,
                            save_as_pdf: bool = True) -> None:
        """
        Create ChArUco board
            :param min_margin_mm: minimum margin in mm
            :param paper_size: paper size (width, height)
            :param dpi: dots per inch for printer specs.
            :param orientation: 'landscape' or 'portrait' for board orientation
            :param save_as_pdf: boolean to save image as pdf or not
        """
        print('{} squares: {} mm'.format(self.squares, tuple(sq * self.marker_size[0] for sq in self.squares)))
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
            fig_width_in = round(paper_size[0] / 25.4, 1)
            fig_height_in = round(paper_size[1] / 25.4, 1)

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
        self.board_image = canvas

    def get_charuco_image(self) -> np.ndarray:
        """
        Function to get ChArUco board image to print
            :return ChArUco board image to print
        """
        return self.board_image

    def view_charuco_board(self):
        """
        View Charuco board image
        """
        cv2.namedWindow('Charuco', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Charuco', self.board_image.shape[1] // 4, self.board_image.shape[0] // 4)
        cv2.imshow('Charuco', self.board_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_image(self, images_path: str, visualize: bool = False) -> None:
        """
        Function to load image from given path.
            :param image_path: path to image
            :param visualize: boolean, if True image will be displayed
        """
        if os.path.exists(images_path):
            self.images_path = images_path
            images_left = sorted(os.listdir(os.path.join(images_path, 'left')))
            images_right = sorted(os.listdir(os.path.join(images_path, 'right')))
            if len(images_left) != len(images_right):
                raise Exception('Number of images do not match')

            print('Found {} pair of images on provided path'.format(len(images_left)))

            # Read images and create a list of it.
            left_img = [cv2.imread(os.path.join(images_path, 'left', str(img_name)), cv2.IMREAD_GRAYSCALE)
                        for img_name in images_left]
            right_img = [cv2.imread(os.path.join(images_path, 'right', str(img_name)), cv2.IMREAD_GRAYSCALE)
                         for img_name in images_right]

            # Convert to numpy array
            self.images_left = np.stack(left_img, axis=-1).astype(np.uint8)  # Convert to uint8
            self.images_right = np.stack(right_img, axis=-1).astype(np.uint8)  # Convert to uint8
        if visualize:
            for k in range(self.images_left.shape[2]):
                concat = np.concatenate((self.images_left[:, :, k], self.images_right[:, :, k]), axis=1)
                cv2.namedWindow('Calibration images', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Calibration images', concat.shape[1] // 2, concat.shape[0] // 2)
                cv2.imshow('Calibration images', concat)
                cv2.waitKey(0)
            cv2.destroyWindow('Calibration images')

    def detect_charuco_board(self, camera: str, visualize: bool = False) -> None:
        """
        Function to detect charuco board in given camera.
            :param camera: camera name to calibrate
            :param visualize: If true show the detected markers
        """
        if camera == 'left':
            images = self.images_left
        elif camera == 'right':
            images = self.images_right
        else:
            raise Exception('Camera must be either left or right')

        for i in range(images.shape[2]):
            corners, ids, _ = cv2.aruco.detectMarkers(image=images[:, :, i], dictionary=self.aruco_dict)
            bgr_img = cv2.cvtColor(images[:, :, i], cv2.COLOR_GRAY2BGR)

            if ids is None:
                print('Not detected markers at image {} from {} camera'.format(i, camera))
                continue
            elif len(ids) > 0:
                response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=corners,
                                                                                             markerIds=ids,
                                                                                             image=images[:, :, i],
                                                                                             board=self.charuco_board)
                # If true will show detected markers and corners images
        
                if visualize and charuco_corners is not None:
                    img_marker = cv2.aruco.drawDetectedMarkers(image=bgr_img.copy(), corners=corners, ids=ids)
                    img_corners = cv2.aruco.drawDetectedCornersCharuco(image=bgr_img.copy(),
                                                                       charucoCorners=charuco_corners,
                                                                       charucoIds=charuco_ids)
                    concat = np.hstack((img_marker, img_corners))
                    cv2.namedWindow('Detected image {} - {}'.format(i, camera), cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Detected image {} - {}'.format(i, camera), concat.shape[1] // 2,
                                     concat.shape[0] // 2)
                    cv2.imshow('Detected image {} - {}'.format(i, camera), concat)
                    cv2.waitKey(0)
                    cv2.destroyWindow('Detected image {} - {}'.format(i, camera))
                # Save charuco corners id for first image as reference
                if i == 0:
                    self.references_ids = set(charuco_ids.flatten())
                if response and charuco_corners is not None:
                    # filtered_corners = []
                    # filtered_ids = []
                    # for corners, corner_id in zip(charuco_corners, charuco_ids):
                    #     if corner_id[0] in self.references_ids:
                    #         filtered_corners.append(corners)
                    #         filtered_ids.append(corner_id)
                    if len(charuco_corners) > 10:
                        self.all_corners[camera].append(np.array(charuco_corners, dtype=np.float32))
                        self.all_ids[camera].append(np.array(charuco_ids, dtype=np.int32))

    def calibration_camera(self, camera: str) -> None:
        """
        Calibrate camera using charuco board
            :param camera: camera name to calibrate
        """
        if len(self.all_corners[camera]) > 0:
            kk, kc = None, None
            ret, kk, kc, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=self.all_corners[camera],
                charucoIds=self.all_ids[camera],
                board=self.charuco_board,
                cameraMatrix=kk,
                distCoeffs=kc,
                imageSize=(self.images_left.shape[1], self.images_left.shape[0]))
            
            if camera not in self.camera_params:
                self.camera_params[camera] = {}

            self.camera_params[camera]['kk'] = kk
            self.camera_params[camera]['kc'] = kc
            self.camera_params[camera]['r'], _ = cv2.Rodrigues(rvecs[0])
            self.camera_params[camera]['t'] = tvecs[0]
            # self.camera_params[camera]['std_kk'] = std_kk
            # self.camera_params[camera]['std_kc'] = std_kc

    def stereo_calibration(self) -> None:
        """
        Stereo calibration using left and right camera
        """
        matched_object_points = []
        matched_corners_left = []
        matched_corners_right = []
        for i in range(min(len(self.all_corners['left']), len(self.all_corners['right']))):
            # Ensure matching ids in both left and right images
            common_ids = np.intersect1d(self.all_ids['left'][i], self.all_ids['right'][i])
            if len(common_ids) > 0:
                indices_left = np.isin(self.all_ids['left'][i], common_ids).flatten()
                indices_right = np.isin(self.all_ids['right'][i], common_ids).flatten()

                matched_object_points.append(self.charuco_board.getChessboardCorners()[common_ids, :])
                matched_corners_left.append(self.all_corners['left'][i][indices_left])
                matched_corners_right.append(self.all_corners['right'][i][indices_right])

        # Now use matched corners to perform stereo calibration
        if matched_corners_left and matched_corners_right:
            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objectPoints=matched_object_points,
                                                            imagePoints1=matched_corners_left,
                                                            imagePoints2=matched_corners_right,
                                                            cameraMatrix1=self.camera_params['left']['kk'],
                                                            distCoeffs1=self.camera_params['left']['kc'],
                                                            cameraMatrix2=self.camera_params['right']['kk'],
                                                            distCoeffs2=self.camera_params['right']['kc'],
                                                            imageSize=(self.images_left.shape[1], self.images_left.shape[0]),
                                                            flags=cv2.CALIB_FIX_INTRINSIC)
    
        print('Stereo calibration done')
        print('R: {}'.format(R))
        print('T: {}'.format(T))
        print('E: {}'.format(E))
        print('F: {}'.format(F))
        # Store on Camera parameters
        self.camera_params['stereo']['R'] = R
        self.camera_params['stereo']['T'] = T
        self.camera_params['stereo']['E'] = E
        self.camera_params['stereo']['F'] = F

    def show_undistorted_image(self, camera: str, images_list: list = None) -> None:
        """
        Show undistorted image from calibrated process
            :param camera: Left or Right camera
            :param images_list: list of images to show
            :return: Visualization of distorted and undistorted image
        """
        kc = self.camera_params[camera]['kc']
        kk = self.camera_params[camera]['kk']

        # Verify camera variable input to define images to show
        if camera == 'left':
            images = self.images_left
        elif camera == 'right':
            images = self.images_right
        else:
            raise Exception('Camera must be either left or right')

        # Check images_list
        if images_list is None:
            print("Show all undistort all images")
            images_list = np.arange(images.shape[2])

        for k in images_list:
            undist_img = cv2.undistort(images[:, :, k], kk, kc, None, None)
            concat = np.concatenate((images[:, :, k], undist_img), axis=1)
            cv2.namedWindow('Undistorted image {} - {}'.format(k, camera), cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Undistorted image {} - {}'.format(k, camera),
                             concat.shape[1] // 2, concat.shape[0] // 2)
            cv2.imshow('Undistorted image {} - {}'.format(k, camera), concat)
            cv2.waitKey(0)
            cv2.destroyWindow('Undistorted image {} - {}'.format(k, camera))

    def homogeneous_matrix(self, camera: str, visualize: bool = False) -> None:

        kk = self.camera_params[camera]['kk']
        kc = self.camera_params[camera]['kc']
        # Verify camera variable input to define images to show
        if camera == 'left':
            images = self.images_left
        elif camera == 'right':
            images = self.images_right
        else:
            raise Exception('Camera must be either left or right')
        w, h = images.shape[1], images.shape[0]

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(kk, kc, (w, h), alpha=1)

        # Generate some points in the original image
        original_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

        # Undistort points using both the original and the new camera matrix
        undistorted_points = cv2.undistortPoints(original_points, kk, kc, P=new_camera_matrix)
        undistorted_points = undistorted_points.reshape(-1, 2)
        original_points = original_points.reshape(-1, 2)

        # Find the homography matrix from original to new undistorted plane
        H, _ = cv2.findHomography(original_points.reshape(-1, 2), undistorted_points)

        print("Homography matrix (Original to New Plane) from {} left:\n {}".format(camera, H))
        self.camera_params[camera]['H'] = H

        if visualize:
            # Apply the homography to warp the original image to the new plane
            for k in range(images.shape[2]):
                warped_image = cv2.warpPerspective(images[:, :, k], H, (w, h))
                # Draw a grid on the original image

                # Display the grid-transformed image
                concat = np.concatenate((images[:, :, k], warped_image), axis=1)
                cv2.namedWindow('Warped Image with Grid {} - {}'.format(k, camera), cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Warped Image with Grid {} - {}'.format(k, camera),
                                 concat.shape[1] // 2, concat.shape[0] // 2)
                cv2.imshow('Warped Image with Grid {} - {}'.format(k, camera), concat)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def save_camera_params(self, path: str) -> None:
        """
        Save camera parameters to a file
            :param path: path to save camera parameters
        """
        def format_matrix(matrix):
            return '\n'.join(['  - [' + ', '.join(f'{item:.4e}' for item in row) + ']' for row in matrix])

        def format_vector(vector):
            return '[' + ', '.join(f'{item:.4e}' for item in vector) + ']'

        with open(path, 'w') as f:
            f.write('# Left and right Charuco calibration \n #{}\n'.format(self.images_path))
            f.write('\ncamera_matrix_left:\n')
            f.write(format_matrix(self.camera_params['left']['kk']) + '\n')
            f.write('\ndist_coeffs_left: ' + format_vector(self.camera_params['left']['kc'].flatten()) + '\n')
            f.write('\nrot_matrix_left:\n')
            f.write(format_matrix(self.camera_params['left']['r']) + '\n')
            f.write('\nt_left: ' + format_vector(self.camera_params['left']['t'].flatten()) + '\n')
            f.write('\ncamera_matrix_right:\n')
            f.write(format_matrix(self.camera_params['right']['kk']) + '\n')
            f.write('\ndist_coeffs_right: ' + format_vector(self.camera_params['right']['kc'].flatten()) + '\n')
            f.write('\nrot_matrix_right:\n')
            f.write(format_matrix(self.camera_params['right']['r']) + '\n')
            f.write('\nt_right: ' + format_vector(self.camera_params['right']['t'].flatten()) + '\n')
            f.write('\nR:\n')
            f.write(format_matrix(self.camera_params['stereo']['R']) + '\n')
            f.write('\nT: ' + format_vector(self.camera_params['stereo']['T'].flatten()) + '\n')