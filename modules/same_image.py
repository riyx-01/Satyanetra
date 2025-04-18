import cv2
import numpy as np

def mse(imageA, imageB):
    # Mean Squared Error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(img1_path, img2_path):
    imageA = cv2.imread(img1_path)
    imageB = cv2.imread(img2_path)

    imageA = cv2.resize(imageA, (256, 256))
    imageB = cv2.resize(imageB, (256, 256))

    error = mse(imageA, imageB)
    return error
