import cv2 as cv
import numpy as np

# #function to generate events from last x frames (depending on buffer size) to determine background -> avoid neural net
# #c is brightness change threshold, e is number of events (after normalization) needed to be considered bg

def generateHistogram(buffer, c, e, neighborhood):
    #gaussain blur
    # blurred_buffer = [cv.GaussianBlur(frame, (5, 5), 1.0) for frame in buffer]
    # changeMap = np.zeros_like(blurred_buffer[0], dtype=np.uint8)
    # for i in range(len(blurred_buffer) - 1):
    #     diff = cv.absdiff(blurred_buffer[i], blurred_buffer[i + 1])
    #     changeMap += (diff > c).astype(np.uint8)

    changeMap = np.zeros_like(buffer[0], dtype=np.uint8)
    if len(buffer) < 2:
    # Not enough frames, return empty masks
        h, w = buffer[0].shape
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    for i in range(len(buffer) - 1):
        diff = cv.absdiff(buffer[i], buffer[i + 1])
        changeMap += (diff > c).astype(np.uint8)
    changeNormalized = cv.normalize(changeMap, None, 0, 255, cv.NORM_MINMAX) #normalize to 0-255
    mask = 255 - changeNormalized  #background/stabler pixels are closer to 255
    _, binmask = cv.threshold(mask, e, 255, cv.THRESH_BINARY)

    kernel_size = 2 * neighborhood + 1
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    strict_mask = cv.erode(binmask, kernel)

    return strict_mask, changeNormalized