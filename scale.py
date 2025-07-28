import cv2 as cv
import numpy as np

def compute_px_to_cm(video_path, params):
    # Open video capture
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read first frame from video.")
        return

    img = frame

    # # Resize large images for easier processing (optional)
    # max_dim = 1000
    # scale_percent = min(max_dim / max(img.shape[0], img.shape[1]), 1.0)
    # img = cv.resize(img, (int(img.shape[1] * scale_percent), int(img.shape[0] * scale_percent)))

    # Convert to grayscale and blur
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_length_px = 0
    ruler_contour = None

    for cnt in contours:
        # Approximate contour
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # Filter rectangular contours (4 sides)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

            if aspect_ratio > 5:  # Heuristic: ruler is long rectangle
                length_px = max(w, h)
                if length_px > max_length_px:
                    max_length_px = length_px
                    ruler_contour = approx

    # Display result
    if ruler_contour is not None:
        scale_px_to_cm = params['ruler_length'] / max_length_px
        print(f"Detected ruler length: {max_length_px:.2f} px")
        print(f"Scale: {scale_px_to_cm:.5f} cm/px")
        return scale_px_to_cm
    else:
        print("No ruler-like contour detected.")

