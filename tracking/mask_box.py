import numpy as np
def genBox(x1, y1, x2, y2, x3, y3, x4, y4, buffer):
    mask = np.zeros_like(buffer[0], dtype=np.uint8)
    for i in range(x1, x4 + 1):
        for j in range(y1, y4 + 1):
            mask[j, i] = 255
    return mask