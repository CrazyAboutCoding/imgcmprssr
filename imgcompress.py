import numpy as np
import cv2

def compress_image(image_path, k, quality):
    image = cv2.imread(image_path)

    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    compressed_image = centers[labels.flatten()]
    compressed_image = compressed_image.reshape(image.shape)

    compressed_image_path = f"compressed_kmeans_{k}_quality_{quality}.jpg"
    cv2.imwrite(compressed_image_path, compressed_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    return compressed_image, compressed_image_path

compressed_img, compressed_img_path = compress_image('assets/Screenshot 2024-03-16 125033.png', k=256, quality=90)

cv2.imshow('Original Image', cv2.imread('assets/Screenshot 2024-03-16 125033.png'))
cv2.imshow('Compressed Image', compressed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()