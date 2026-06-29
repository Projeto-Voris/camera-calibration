import numpy as np
import cv2
import os

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# cv2.namedWindow('left', cv2.WINDOW_NORMAL)
# cv2.namedWindow('right', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('left', 640, 480)
# cv2.resizeWindow('right', 640, 480)
path = '/home/daniel/Pictures/20260610_1654_stereo_images'

left_images = sorted(os.listdir(os.path.join(path, 'left')))
right_images = sorted(os.listdir(os.path.join(path, 'right')))
print(left_images)
print(right_images)

for left_image, right_image in zip(left_images, right_images):

    print(f"Processing {left_image} and {right_image}...")

    left_img = cv2.imread(os.path.join(path, 'left', left_image), 1)
    right_img = cv2.imread(os.path.join(path, 'right', right_image), 1)

    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2LAB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2LAB)

    

    left_clahe = clahe.apply(left_img[:,:,0])
    right_clahe = clahe.apply(right_img[:,:,0])

    left_out = cv2.cvtColor(cv2.merge((left_clahe, left_img[:,:,1], left_img[:,:,2])), cv2.COLOR_LAB2BGR)
    right_out = cv2.cvtColor(cv2.merge((right_clahe, right_img[:,:,1], right_img[:,:,2])), cv2.COLOR_LAB2BGR)


    # cv2.imshow('left', left_out)
    # cv2.imshow('right', right_out)

    cv2.imwrite(os.path.join(path, 'left', left_image.split('.')[0] + '_clahe.png'), left_out)
    cv2.imwrite(os.path.join(path, 'right', right_image.split('.')[0] + '_clahe.png'), right_out)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break