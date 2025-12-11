import cv2
import numpy as np
import glob

# Define os parâmetros do tabuleiro de xadrez
CHECKERBOARD = (15,10)
SQUARE_SIZE = 0.25  # metros

# Critérios de término para cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepara os pontos do mundo real
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays para armazenar pontos 3D e 2D
objpoints = []
imgpoints_left = []
imgpoints_right = []

# Caminhos das imagens
images_left = sorted(glob.glob('/home/daniel/Pictures/image_logs/SM2/20250522_1_2/left/*.png'))
images_right = sorted(glob.glob('/home/daniel/Pictures/image_logs/SM2/20250522_1_2/right/*.png'))

image_size  = None

for img_left_path, img_right_path in zip(images_left, images_right):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = (int(gray_left.shape[1]), int(gray_left.shape[0]))

    ret_l, corners_l = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp.reshape(-1, 1, 3).astype(np.float32))

        corners2_l = cv2.cornerSubPix(gray_left, corners_l, (11,11), (-1,-1), criteria)
        corners2_r = cv2.cornerSubPix(gray_right, corners_r, (11,11), (-1,-1), criteria)

        imgpoints_left.append(corners2_l.astype(np.float32))
        imgpoints_right.append(corners2_r.astype(np.float32))


# Estimativa da calibração estéreo
flags = 0
flags |= cv2.omnidir.CALIB_FIX_K1
flags |= cv2.omnidir.CALIB_FIX_K2

K1 = np.eye(3, dtype=np.float64)
D1 = np.zeros((4, 1), dtype=np.float64)
xi1 = np.array([0.0], dtype=np.float64)

K2 = np.eye(3, dtype=np.float64)
D2 = np.zeros((4, 1), dtype=np.float64)
xi2 = np.array([0.0], dtype=np.float64)

R = np.eye(3, dtype=np.float64)
t = np.zeros((3, 1), dtype=np.float64)
image_size_cv = (int(image_size[0]), int(image_size[1]))  # garantir que seja tupla de dois inteiros

print(f"{len(objpoints)=}, {len(imgpoints_left)=}, {len(imgpoints_right)=}")
print(f"{objpoints[0].shape=}, {imgpoints_left[0].shape=}, {imgpoints_right[0].shape=}")
print(f"{image_size_cv=}")

rms, K1, xi1, D1, K2, xi2, D2, R, t, rvecsL, tvecsL, idx = cv2.omnidir.stereoCalibrate(
    objectPoints=objpoints,
    imagePoints1=imgpoints_left,
    imagePoints2=imgpoints_right,
    imageSize1=image_size_cv,
    imageSize2=image_size_cv,
    K1=K1,
    xi1=xi1,
    D1=D1,
    K2=K2,
    xi2=xi2,
    D2=D2,
    flags=flags,
    criteria=criteria
)


print("RMS:", rms)
print("K1:", K1)
print("xi1:", xi1)
print("D1:", D1)
print("K2:", K2)
print("xi2:", xi2)
print("D2:", D2)
print("R:", R)
print("T:", t)