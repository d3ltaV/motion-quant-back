import cv2 as cv
import numpy as np
import time
from tracking.bg_segmentation import generateHistogram

def run_sparse(vars, p):
    cap = vars["cap"]
    frame_buffer = vars["frame_buffer"]
    prev_gray = vars["prev_gray"]
    background_mask = vars["background_mask"]
    foreground_mask = vars["foreground_mask"]
    prev_leaf_pts = vars["prev_leaf_pts"]
    prev_bg_pts = vars["prev_bg_pts"]
    mask = vars["mask"]
    alpha = vars["alpha"]
    max_motion = vars["max_motion"]
    window_size = vars["window_size"]
    motion_window = vars["motion_window"]
    smoothed_motion = vars["smoothed_motion"]
    total_motion = vars["total_motion"]
    total_cam_motion = vars["total_cam_motion"]
    
    low_motion_threshold = vars["low_motion_threshold"] #for foreground points -> not enough mvt
    high_motion_threshold = vars["high_motion_threshold"] #for background points -> too much mvt
    low_motion_fraction = vars["low_motion_fraction"]
    high_motion_fraction = vars["high_motion_fraction"]

    bg_frame_cnt = vars["bg_frame_cnt"] #frame cnt for when too many bg points move
    f_frame_cnt = vars["f_frame_cnt"] #frame cnt for when not enough foreground points move

    font = vars["font"]
    font_scale = vars["font_scale"]
    text_thickness = vars["text_thickness"]
    text_color = vars["text_color"]
    bar_color = vars["bar_color"]
    text_width = vars["text_width"]
    text_height = vars["text_height"]
    reserved_text_size = vars["reserved_text_size"]
    out = vars["out"]
    feature_params = vars["feature_params"]
    lk_params = vars["lk_params"]
    flow_color = vars["flow_color"]
    point_color = vars["point_color"]
    bg_point_color = vars["bg_point_color"]
    bg_status = vars["bg_status"]
    leaf_status = vars["leaf_status"]
    
    c = p["brightness_thresh"]
    e = p["event_thresh"]
    n = p["neighborhood"]

    active_scale = p.get("active_scale", 0.08152)  # use from params or default
    currNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        return False

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_buffer.append(gray.copy())

    #background mask
    if (currNum > 4) and ((bg_status is None) or (np.count_nonzero(bg_status == 1) <= 10) or (leaf_status is None) or (np.count_nonzero(leaf_status == 1) < 10)
    or (bg_frame_cnt >= 5) or (f_frame_cnt >= 5)): #if there's no bg movement for many frame
        bg_frame_cnt = 0
        f_frame_cnt = 0
        background_mask, bgNormalized = generateHistogram(list(frame_buffer), c, e, n)
        foreground_mask = cv.bitwise_not(background_mask)

    #Shi Tomasi points for foreground and background

    #if you want to keep tracking the same points -> albeit less accurate
    # if prev_leaf_pts is None or len(prev_leaf_pts) < 1:      
    #     prev_leaf_pts = cv.goodFeaturesToTrack(prev_gray, mask=foreground_mask, **feature_params)
    #     if prev_leaf_pts is None:
    #         prev_leaf_pts = np.array([]).reshape(-1, 1, 2)
    prev_leaf_pts = cv.goodFeaturesToTrack(prev_gray, mask=foreground_mask, **feature_params) #shi tomasi every frame, comment out if top -> use foreground_mask = None for 1st week vid
    if prev_leaf_pts is None:
        prev_leaf_pts = np.array([]).reshape(-1, 1, 2)

    prev_bg_pts = cv.goodFeaturesToTrack(prev_gray, mask=background_mask, **feature_params) #shi tomasi
    if prev_bg_pts is None:
        prev_bg_pts = np.array([]).reshape(-1, 1, 2)

    # Optical flow tracking
    if prev_leaf_pts.size > 0:
        leaf_next_pts, leaf_status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_leaf_pts, None, **lk_params)
        if leaf_next_pts is None or leaf_status is None:
            leaf_next_pts = np.array([]).reshape(-1, 1, 2)
            leaf_status = np.array([], dtype=np.uint8)
    else:
        leaf_next_pts = np.array([]).reshape(-1, 1, 2)
        leaf_status = np.array([], dtype=np.uint8)

    if prev_bg_pts.size > 0:
        bg_next_pts, bg_status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_bg_pts, None, **lk_params)
        if bg_next_pts is None or bg_status is None:
            bg_next_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)
            bg_status = np.array([], dtype=np.uint8)
    else:
        bg_next_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)
        bg_status = np.array([], dtype=np.uint8)

    #draw bg motion vec
    if (bg_next_pts is not None and bg_status is not None and prev_bg_pts is not None and 
        bg_next_pts.size > 0 and bg_status.size > 0 and prev_bg_pts.size > 0):
        bg_old = prev_bg_pts[bg_status == 1].astype(np.float32)
        bg_new = bg_next_pts[bg_status == 1].astype(np.float32)
        for (new_bg, old_bg) in zip(bg_new.astype(int), bg_old.astype(int)):
            x1, y1 = new_bg.ravel()
            x2, y2 = old_bg.ravel()
            mask = cv.line(mask, (x1, y1), (x2, y2), (0, 140, 255), 1)
            frame = cv.circle(frame, (x2, y2), 4, (255, 255, 0), -1)
    
    #check for too much background motion -> update mask
    if bg_next_pts is not None and bg_status is not None and prev_bg_pts is not None:
        b_new = bg_next_pts[bg_status == 1].astype(int)
        b_old = prev_bg_pts[bg_status == 1].astype(int)
        if len(b_old) > 0:
            displacements = np.linalg.norm(b_new - b_old, axis=1)  # Euclidean distance per point
            fraction_above_threshold = np.count_nonzero(displacements > high_motion_threshold) / len(displacements)

            if fraction_above_threshold > high_motion_fraction:
                bg_frame_cnt += 1
            else:
                bg_frame_cnt = 0  # Reset if there's movement

    #check for too little foreground motion -> update mask
    if leaf_next_pts is not None and leaf_status is not None and prev_leaf_pts is not None:
        l_new = leaf_next_pts[leaf_status == 1].astype(int)
        l_old = prev_leaf_pts[leaf_status == 1].astype(int)
        if len(l_old) > 0:
            displacements = np.linalg.norm(l_new - l_old, axis=1)  # Euclidean distance per point
            fraction_below_threshold = np.count_nonzero(displacements < low_motion_threshold) / len(displacements)

            if fraction_below_threshold > low_motion_fraction:
                f_frame_cnt += 1
            else:
                f_frame_cnt = 0  # Reset if there's movement
                
    total_motion_frame = 0.0
    if leaf_next_pts is not None and leaf_status is not None:
        good_old = prev_leaf_pts[leaf_status == 1].astype(np.float32)
        good_new = leaf_next_pts[leaf_status == 1].astype(np.float32)
        for (new, old) in zip(good_new, good_old):
            leaf_disp = new.reshape(2) - old.reshape(2)  #vec
            motion_real = np.linalg.norm(leaf_disp) * active_scale
            if not np.isnan(motion_real):
                total_motion_frame += motion_real
            # motion_pixels = np.linalg.norm([a - c, b - d])
            # motion_real = motion_pixels * active_scale
            # total_motion_frame += motion_real
            a, b = new.astype(int).ravel()
            c, d = old.astype(int).ravel()
            mask = cv.line(mask, (a, b), (c, d), flow_color, 1)
            frame = cv.circle(frame, (a, b), 3, point_color, -1)

    # Update motion smoothing
    motion_window.append(total_motion_frame)
    if len(motion_window) > window_size:
        motion_window.pop(0)
    
    if len(motion_window) > 0:
        moving_avg = sum(motion_window) / len(motion_window)
    else:
        moving_avg = 0.0

    smoothed_motion = alpha * moving_avg + (1 - alpha) * smoothed_motion
    if np.isnan(smoothed_motion):
        smoothed_motion = 0
    total_motion += smoothed_motion
    output = cv.add(frame, mask)

    text = f"Smoothed motion: {smoothed_motion:.4f}cm"
    text_x = output.shape[1] - text_width - 20
    text_y = 30
    cv.putText(output, text, (text_x, text_y), font, font_scale, text_color, text_thickness)
    text2 = f"Total motion: {total_motion:.4f}cm"
    text2_x = output.shape[1] - text_width - 20
    text2_y = 90
    cv.putText(output, text2, (text2_x, text2_y), font, font_scale, text_color, text_thickness)

    # Draw motion bar
    bar_width = 200
    bar_height = 20
    bar_x = text_x
    bar_y = text_y + bar_height
    normalized_motion = min(smoothed_motion / max_motion, 1.0)
    if np.isnan(smoothed_motion):
        smoothed_motion = 0.0
        filled_width = 0
    else:
        filled_width = int(bar_width * normalized_motion)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 1)
    
    # Count the number of successfully tracked points for each type
    leaf_count = int(np.count_nonzero(leaf_status == 1)) if leaf_status is not None else 0
    bg_count = int(np.count_nonzero(bg_status == 1)) if bg_status is not None else 0

    print(f"Frame {currNum}: Leaf points = {leaf_count}, Background points = {bg_count}")

    # Update previous frame and points
    if (leaf_next_pts is not None) and (len(leaf_next_pts) > 0):
        prev_leaf_pts = leaf_next_pts.reshape(-1, 1, 2)
    if (bg_next_pts is not None) and (len(bg_next_pts) > 0):
        prev_bg_pts = bg_next_pts.reshape(-1, 1, 2)

    #draw binmask
    if background_mask is not None:
        mask_color = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)
        mask_small = cv.resize(mask_color, (160, 120))
        x_offset = 20
        y_offset = output.shape[0] - 20 - mask_small.shape[0]
        output[y_offset:y_offset + mask_small.shape[0], x_offset:x_offset + mask_small.shape[1]] = mask_small

    vars["prev_gray"] = gray.copy()
    vars["prev_leaf_pts"] = leaf_next_pts.reshape(-1, 1, 2) if leaf_next_pts.size > 0 else np.array([]).reshape(-1, 1, 2)
    vars["prev_bg_pts"] = bg_next_pts.reshape(-1, 1, 2) if bg_next_pts.size > 0 else np.array([]).reshape(-1, 1, 2)
    vars["bg_status"] = bg_status
    vars["leaf_status"] = leaf_status
    vars["smoothed_motion"] = smoothed_motion
    vars["total_motion"] = total_motion
    vars["total_cam_motion"] = total_cam_motion
    vars["bg_frame_cnt"]= bg_frame_cnt
    vars["f_frame_cnt"] = f_frame_cnt
    vars["background_mask"] = background_mask
    vars["foreground_mask"] = foreground_mask
    vars["mask"] = mask

    out.write(output)
    return True
