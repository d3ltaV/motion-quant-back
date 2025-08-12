import cv2 as cv
import numpy as np
import time

def run_bounding_sparse(vars, p): 
    cap = vars["cap"]
    prev_gray = vars["prev_gray"]
    prev = vars["prev_leaf_pts"]
    box_foreground_mask = vars["box_foreground_mask"]
    box_background_mask = vars["box_background_mask"]
    mask = vars["mask"]
    alpha = vars["alpha"]
    max_motion = vars["max_motion"]
    window_size = vars["window_size"]
    motion_window = vars["motion_window"]
    smoothed_motion = vars["smoothed_motion"]
    total_motion = vars["total_motion"]
    total_cam_motion = vars["total_cam_motion"]
    font = vars["font"]
    font_scale = vars["font_scale"]
    text_thickness = vars["text_thickness"]
    text_color = vars["text_color"]
    bar_color = vars["bar_color"]
    text_width = vars["text_width"]
    out = vars["out"]
    feature_params = vars["feature_params"]
    lk_params = vars["lk_params"]
    flow_color = vars["flow_color"]
    point_color = vars["point_color"]
    bg_point_color = vars["bg_point_color"]
    leaf_status = vars["leaf_status"]
    active_scale = p.get("active_scale", 0.08152)  # use from params or default
    frame_start = time.time()
    currNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))

    ret, frame = cap.read()
    if not ret:
        return False

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    prev = cv.goodFeaturesToTrack(prev_gray, mask=box_foreground_mask, **feature_params)
    next_pts, leaf_status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    if prev is None:
        print("No good features to track.")
        return True

    if next_pts is None:
        print("Optical flow failed.")
        return True
    
    good_old = prev[leaf_status == 1].astype(int)
    good_new = next_pts[leaf_status == 1].astype(int)

    total_motion_frame = 0
    for (new, old) in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), flow_color, 1)
        frame = cv.circle(frame, (a, b), 3, point_color, -1)

        # Calculate motion in px, convert to cm
        motion_pixels = np.linalg.norm([a - c, b - d])
        motion_real = motion_pixels * active_scale
        total_motion_frame += motion_real

    # Update moving window with new raw total_motion
    motion_window.append(total_motion_frame)
    if len(motion_window) > window_size:
        motion_window.pop(0)

    # Compute moving average over the window
    moving_avg = sum(motion_window) / len(motion_window)

    # Apply exponential smoothing on moving average
    smoothed_motion = alpha * moving_avg + (1 - alpha) * smoothed_motion
    total_motion += smoothed_motion

    # Overlay tracking
    output = cv.add(frame, mask)

    # Text (fixed position based on reserved size)
    text = f"Smoothed motion: {smoothed_motion:.4f}cm"
    text_x = output.shape[1] - text_width - 20
    text_y = 30
    cv.putText(output, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

    # Text 2: Total motion
    text2 = f"Total motion: {total_motion:.4f}cm"
    text2_x = output.shape[1] - text_width - 20
    text2_y = 90
    cv.putText(output, text2, (text2_x, text2_y), font, font_scale, text_color, text_thickness)

    # Motion bar below text
    bar_width = 200
    bar_height = 20
    bar_x = text_x
    bar_y = text_y + bar_height  # Padding

    normalized_motion = min(smoothed_motion / max_motion, 1.0)
    filled_width = int(bar_width * normalized_motion)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 1)

    frame_end = time.time()
    frame_time = (frame_end - frame_start) * 1000  # 1000 for ms
    
    #draw binmask
    if box_foreground_mask is not None:
        mask_color = cv.cvtColor(box_foreground_mask, cv.COLOR_GRAY2BGR)
        mask_small = cv.resize(mask_color, (160, 120))
        x_offset = 20
        y_offset = output.shape[0] - 20 - mask_small.shape[0]
        output[y_offset:y_offset + mask_small.shape[0], x_offset:x_offset + mask_small.shape[1]] = mask_small
        
    out.write(output)
    
    leaf_next_pts = good_new if good_new.size > 0 else np.array([]).reshape(-1, 1, 2)
    leaf_count = int(np.count_nonzero(leaf_status == 1)) if leaf_status is not None else 0

    print(f"Frame {currNum}: Leaf points = {leaf_count}")

    #update variables properly
    vars["prev_gray"] = gray.copy()
    vars["prev_leaf_pts"] = leaf_next_pts.reshape(-1, 1, 2) if leaf_next_pts.size > 0 else np.array([]).reshape(-1, 1, 2)
    vars["leaf_status"] = leaf_status
    vars["smoothed_motion"] = smoothed_motion
    vars["total_motion"] = total_motion
    vars["total_cam_motion"] = total_cam_motion
    
    return True

