import cv2 as cv
import numpy as np
from collections import deque
from tracking.bg_segmentation import generateHistogram
from tracking.mask_box import genBox
from tracking.algos.sparse import run_sparse
from tracking.algos.dense import run_dense
from tracking.algos.no_mvt_sparse import run_no_mvt_sparse
from tracking.algos.bounding_sparse import run_bounding_sparse
import os

default_params = {
    "active_scale": 0.08152,
    "brightness_thresh": 0.15,
    "event_thresh": 0.5,
    "neighborhood": 50,
    "bgSeg": "Sparse",
    "adjust_for_resolution": False
}

#Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=140, qualityLevel=0.15, minDistance=1, blockSize=7)

#Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Colors and font
flow_color = (0, 255, 0)
point_color = (0, 0, 255)
bg_point_color = (255, 200, 100)
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
text_thickness = 2
text_color = (255, 255, 255)
bar_color = (0, 0, 255)

def adjust_scale(params):
    if (params['resolution_width'] != params['scale_resolution_width'] or 
        params['resolution_height'] != params['scale_resolution_height']):

        diag_scale = (params['scale_resolution_width'] ** 2 + params['scale_resolution_height'] ** 2) ** 0.5
        diag_input = (params['resolution_width'] ** 2 + params['resolution_height'] ** 2) ** 0.5

        # Scale factor is proportional to change in diagonal
        scale_factor = diag_scale / diag_input
        params['active_scale'] *= scale_factor


def run_motion_quant(video_path, params, output_dir='outputs'):
    p = {**default_params, **params}
    # print("Final Parameters for Motion Quantification:")
    # for k, v in p.items():
    #     print(f"  {k}: {v}")
    if isinstance(p["adjust_for_resolution"], str):
        val = p["adjust_for_resolution"].strip().lower() == "true"
    p["adjust_for_resolution"] = val
    if p["adjust_for_resolution"] == True:
        adjust_scale(p)
    print("Adjusted active_scale:", p["active_scale"])
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(100 / fps) #if vid not here fps = 0 -> error
    # delay = max(1, int(1000 / fps))  # Convert to milliseconds
    
    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    frame_buffer = deque(maxlen=5)
    frame_buffer.append(prev_gray.copy())
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.avi")

    while len(frame_buffer) < 5:
        ret, frame = cap.read()
        if not ret:
            break
        g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_buffer.append(g.copy())
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 1)
    background_mask, bgNormalized = generateHistogram(
        list(frame_buffer),
        p["brightness_thresh"],
        p["event_thresh"],
        p["neighborhood"]
    ) #buffer, brightness change thresh, normalized event count thresh, erode

    foreground_mask = cv.bitwise_not(background_mask)
    
    x1 = 240
    y1 = 1
    x2 = 1
    y2 = 1
    x3 = 1100
    y3 = 600
    x4 = 1
    y4 = 1
    box_foreground_mask = genBox(x1, y1, x2, y2, x3, y3, x4, y4, list(frame_buffer))
    box_background_mask = cv.bitwise_not(box_foreground_mask)

    frame_prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params) #initial shi tomasi to help camera tracking (not recomputed every frame)
    prev_leaf_pts = cv.goodFeaturesToTrack(prev_gray, mask=foreground_mask, **feature_params) #initial shi tomasi for leaves
    prev_bg_pts = cv.goodFeaturesToTrack(prev_gray, mask=background_mask, **feature_params) #initial shi tomasi for bg
    
    mask = np.zeros_like(first_frame) #canvas for drawing optical flow vecs
    bg_next_pts = None
    bg_status = None
    leaf_status = None
    leaf_next_pts = None
    frame_next_pts = None
    frame_pts_status = None

    frame_buffer = deque(maxlen=5)
    frame_buffer.append(prev_gray.copy())
    cap.set(cv.CAP_PROP_POS_FRAMES, 1)

    # Video writer setup (same size as input video)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    #out = cv.VideoWriter('output.avi', fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    out = cv.VideoWriter(output_path, fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not open video writer for output file: {output_path}")
    
    # Exponential smoothing setup
    alpha = 0.1  # Smoothing factor
    max_motion = 200  # For bar scaling

    # Moving window for raw motion values
    window_size = 5
    motion_window = []
    smoothed_motion = 0
    total_motion = 0
    total_cam_motion = 0

    low_motion_threshold = 3 #if less than 3 pixel of mvt per frame
    low_motion_fraction = 0.5
    high_motion_threshold = 1
    high_motion_fraction = 0.5
    bg_frame_cnt = 0 #frames where many frame tracking points don't move
    f_frame_cnt = 0
    # Reserve space for longest expected text
    reserved_text = "Smoothed motion: 200.00cm"
    reserved_text_size, _ = cv.getTextSize(reserved_text, font, font_scale, text_thickness)
    text_width, text_height = reserved_text_size

    vars = {
        "cap": cap,
        "fps": fps,
        "delay": delay,
        "prev_gray": prev_gray,

        "frame_buffer": frame_buffer,
        "background_mask": background_mask,
        "foreground_mask": foreground_mask,
        "box_foreground_mask": box_foreground_mask,
        "box_background_mask": box_background_mask,
        
        "frame_prev_pts": frame_prev_pts,
        "frame_next_pts": frame_next_pts,
        "frame_pts_status": frame_pts_status,
        "prev_leaf_pts": prev_leaf_pts,
        "leaf_status":leaf_status,
        "leaf_next_pts": leaf_next_pts,
        "bg_next_pts": bg_next_pts,
        "prev_bg_pts": prev_bg_pts,
        "bg_status": bg_status,

        "mask": mask,
        "alpha": alpha,
        "max_motion": max_motion,
        "window_size": window_size,
        "motion_window": motion_window,
        "smoothed_motion": smoothed_motion,
        "total_motion": total_motion,
        "total_cam_motion": total_cam_motion,
        "low_motion_threshold": low_motion_threshold,
        "low_motion_fraction": low_motion_fraction,
        "high_motion_threshold": high_motion_threshold,
        "high_motion_fraction": high_motion_fraction,
        "bg_frame_cnt": bg_frame_cnt,
        "f_frame_cnt": f_frame_cnt,
        "font": font,
        "font_scale": font_scale,
        "text_thickness": text_thickness,
        "text_color": text_color,
        "bar_color": bar_color,
        "flow_color": flow_color,
        "point_color": point_color,
        "bg_point_color": bg_point_color,
        "reserved_text_size": reserved_text_size,
        "text_width": text_width,
        "text_height": text_height,
        "fourcc": fourcc,
        "out": out,
        "feature_params": feature_params,
        "lk_params": lk_params,
    }
    try:
        if (params['processing_algo'] == "Sparse with moving camera"):
            while cap.isOpened():
                if not run_sparse(vars, p):
                # if not run_bounding_sparse(vars, p):
                    break
        elif (params['processing_algo'] == "Dense with moving camera"):
            while cap.isOpened():
                if not run_dense(vars, p):
                    break
        elif (params['processing_algo'] == "Sparse without moving camera"):
            while cap.isOpened():
                if not run_no_mvt_sparse(vars, p):
                    break
        elif (params['processing_algo'] == "Bounding"):
            while cap.isOpened():
                if not run_bounding_sparse(vars, p):
                    break

    finally:
        cap.release()
        out.release()
        cv.destroyAllWindows()

    return output_path