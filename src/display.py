import cv2
import numpy as np


def montages(montages, scale=True):
    for image in montages:
        img = np.copy(image)
        if scale:
            img = img / 255.

        cv2.imshow('Montage', img)
        cv2.waitKey(0)


def image(image, scale=True, wait=True):
    img = np.copy(image)
    if scale:
        img = img / 255.
    cv2.imshow('Image', img)

    if wait:
        cv2.waitKey(0)