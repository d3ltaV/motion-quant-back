import cv2 as cv
import numpy as np
#function to generate events from last x frames (depending on buffer size) to determine background -> avoid neural net
#c is brightness change threshold, e is number of events (after normalization) needed to be considered bg\

def generateHistogram(buffer, c_ratio, e_ratio, neighborhood):
    if len(buffer) < 2:
        h, w = buffer[0].shape
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    #denoise with blur
    blurred_buffer = [cv.medianBlur(frame, 5, 1.0) for frame in buffer]

    changeMap = np.zeros_like(blurred_buffer[0], dtype=np.uint8)

    # find global maximum brightness change
    max_diff_value = 0
    diffs = []
    for i in range(len(blurred_buffer) - 1):
        diff = cv.absdiff(blurred_buffer[i], blurred_buffer[i + 1])
        max_diff_value = max(max_diff_value, np.max(diff))
        diffs.append(diff)

    # dynamic based on c_ratio of max_diff_value
    c = max_diff_value * c_ratio

    # compute changeMap using adaptive c
    for diff in diffs:
        changeMap += (diff > c).astype(np.uint8)

    # Normalize to 0-255
    changeNormalized = cv.normalize(changeMap, None, 0, 255, cv.NORM_MINMAX)

    # Dynamic e based on e_ratio of maximum event count
    max_events = np.max(changeMap)
    e = max_events * e_ratio

    # Background/stable pixels are closer to 255
    mask = 255 - changeNormalized
    _, binmask = cv.threshold(mask, e, 255, cv.THRESH_BINARY)

    # Morphological erosion for strict mask
    kernel_size = int(2 * neighborhood + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    strict_mask = cv.erode(binmask, kernel)

    return strict_mask, changeNormalized
