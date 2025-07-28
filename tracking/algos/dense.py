import cv2 as cv
import numpy as np
import time
from tracking.bg_segmentation import generateHistogram

def run_dense(vars, p):
    cap = vars["cap"]
    frame_buffer = vars["frame_buffer"]
    prev_gray = vars["prev_gray"]
    background_mask = vars["background_mask"]
    foreground_mask = vars["foreground_mask"]
    frame_prev_pts = vars["frame_prev_pts"]
    prev_leaf_pts = vars["prev_leaf_pts"]
    mask = vars["mask"]
    alpha = vars["alpha"]
    max_motion = vars["max_motion"]
    window_size = vars["window_size"]
    motion_window = vars["motion_window"]
    smoothed_motion = vars["smoothed_motion"]
    total_motion = vars["total_motion"]
    total_cam_motion = vars["total_cam_motion"]
    low_motion_threshold = vars["low_motion_threshold"]
    low_motion_fraction = vars["low_motion_fraction"]
    frame_cnt = vars["frame_cnt"]
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
    frame_pts_status = vars["frame_pts_status"]
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
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    frame_buffer.append(gray.copy())

    if (currNum > 4) and ((bg_status is None) or (np.count_nonzero(bg_status == 1) <= 10) or (leaf_status is None)
        or (np.count_nonzero(leaf_status == 1) < 10) or (frame_pts_status is None) or (np.count_nonzero(frame_pts_status) < 100)
        or (frame_cnt >= 5)):
        frame_cnt = 0
        frame_prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if frame_prev_pts is None:
            frame_prev_pts = np.array([]).reshape(-1, 1, 2)
        background_mask, bgNormalized = generateHistogram(frame_buffer, 8, 190, 40)
        foreground_mask = cv.bitwise_not(background_mask)

    prev_leaf_pts = cv.goodFeaturesToTrack(prev_gray, mask=foreground_mask, **feature_params)
    if prev_leaf_pts is None:
        prev_leaf_pts = np.array([]).reshape(-1, 1, 2)
    
    avg_bg_motion = np.array([0.0, 0.0], dtype=np.float32) #avg. vector of bg points
    camera_motion_magnitude = 0.0
    
    # Dense optical flow for background motion -> find avg_bg_motion and camera_motion_magnitude
    if prev_gray is not None:
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        if background_mask is not None:
            valid_bg_pixels = background_mask > 0
            bg_flow_x = flow_x[valid_bg_pixels]
            bg_flow_y = flow_y[valid_bg_pixels]
            bg_magnitude = np.sqrt(bg_flow_x**2 + bg_flow_y**2)
            avg_dx = np.mean(bg_flow_x)
            avg_dy = np.mean(bg_flow_y)
            if bg_flow_x.size > 0:
                avg_bg_motion = np.array([np.mean(bg_flow_x), np.mean(bg_flow_y)], dtype=np.float32)
                camera_motion_magnitude = np.mean(bg_magnitude) * active_scale

                h, w = prev_gray.shape
                y_coords, x_coords = np.mgrid[10:h-10:20, 10:w-10:20].reshape(2, -1).astype(int)
                for y, x in zip(y_coords, x_coords):
                    if background_mask[y, x] > 0:
                        dx, dy = flow[y, x]
                        if abs(dx) > 0.5 or abs(dy) > 0.5:
                            end_x, end_y = int(x + dx), int(y + dy)
                            mask = cv.line(mask, (x, y), (end_x, end_y), (0, 140, 255), 1)
                            frame = cv.circle(frame, (x, y), 2, (255, 255, 0), -1)
            print(f"[DENSE FLOW] avg dx: {avg_dx:.3f}, dy: {avg_dy:.3f}, mag: {camera_motion_magnitude:.3f}")

    leaf_next_pts, leaf_status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_leaf_pts, None, **lk_params)
    if leaf_next_pts is None or leaf_status is None:
        leaf_next_pts = np.array([]).reshape(-1, 1, 2)
        leaf_status = np.array([], dtype=np.uint8)

    frame_next_pts, frame_pts_status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, frame_prev_pts, None, **lk_params)
    if frame_next_pts is None or frame_pts_status is None:
        frame_next_pts = np.array([]).reshape(-1, 1, 2)
        frame_pts_status = np.array([], dtype=np.uint8)

    if not np.isnan(camera_motion_magnitude):
        total_cam_motion += camera_motion_magnitude

    if frame_next_pts is not None and frame_pts_status is not None and frame_prev_pts is not None:
        f_old = frame_prev_pts[frame_pts_status == 1].astype(int)
        f_new = frame_next_pts[frame_pts_status == 1].astype(int)
        if len(f_old) > 0:
            displacements = np.linalg.norm(f_new - f_old, axis=1)
            fraction_below_threshold = np.count_nonzero(displacements < low_motion_threshold) / len(displacements)
            if fraction_below_threshold > low_motion_fraction:
                frame_cnt += 1
            else:
                frame_cnt = 0
            for (new, old) in zip(f_old, f_new):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv.circle(frame, (a, b), 4, (255, 0, 225), -1)

    total_motion_frame = 0.0
    if leaf_next_pts is not None and leaf_status is not None:
        good_old = prev_leaf_pts[leaf_status == 1].astype(np.float32)
        good_new = leaf_next_pts[leaf_status == 1].astype(np.float32)
        for (new, old) in zip(good_new, good_old):
            leaf_disp = new.reshape(2) - old.reshape(2)
            corrected_disp = leaf_disp - avg_bg_motion
            motion_real = np.linalg.norm(corrected_disp) * active_scale
            if not np.isnan(motion_real):
                total_motion_frame += motion_real
            a, b = new.astype(int).ravel()
            c, d = old.astype(int).ravel()
            mask = cv.line(mask, (a, b), (c, d), flow_color, 1)
            frame = cv.circle(frame, (a, b), 3, point_color, -1)

    motion_window.append(total_motion_frame)
    if len(motion_window) > window_size:
        motion_window.pop(0)

    moving_avg = sum(motion_window) / len(motion_window)
    smoothed_motion = alpha * moving_avg + (1 - alpha) * smoothed_motion
    if np.isnan(smoothed_motion):
        smoothed_motion = 0
    total_motion += smoothed_motion
    if np.isnan(camera_motion_magnitude):
        camera_motion_magnitude = 0

    output = cv.add(frame, mask)

    text = f"Smoothed motion: {smoothed_motion:.4f}cm"
    text_x = output.shape[1] - text_width - 20
    text_y = 30
    cv.putText(output, text, (text_x, text_y), font, font_scale, text_color, text_thickness)
    text2 = f"Total motion: {total_motion:.4f}cm"
    text2_x = output.shape[1] - text_width - 20
    text2_y = 90
    cv.putText(output, text2, (text2_x, text2_y), font, font_scale, text_color, text_thickness)
    # avg_camera_motion = np.linalg.norm(avg_bg_motion) * active_scale
    text3 = f"Camera motion: {camera_motion_magnitude:.4f}cm"
    text3_x = output.shape[1] - text_width - 20
    text3_y = 150
    cv.putText(output, text3, (text3_x, text3_y), font, font_scale, text_color, text_thickness)
    text4 = f"Total cam motion: {total_cam_motion:.4f}cm"
    text4_x = output.shape[1] - text_width - 20
    text4_y = 210
    cv.putText(output, text4, (text4_x, text4_y), font, font_scale, text_color, text_thickness)

    # Draw motion bar
    bar_width = 200
    bar_height = 20
    bar_x = text_x
    bar_y = text_y + bar_height
    normalized_motion = min(smoothed_motion / max_motion, 1.0)
    if np.isnan(smoothed_motion):
        smoothed_motion = 0.0
    else:
        filled_width = int(bar_width * normalized_motion)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
    cv.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 1)

    bar1_x = text3_x
    bar1_y = text3_y + bar_height
    if np.isnan(camera_motion_magnitude):
        camera_motion_magnitude = 0.0
        norm_cam_motion = 0.0
    else:
        norm_cam_motion = min(camera_motion_magnitude/ max_motion, 1.0)
    cam_filled_width = max(1, int(bar_width * norm_cam_motion)) if norm_cam_motion > 0 else 0
    cv.rectangle(output, (bar1_x, bar1_y), (bar1_x + bar_width, bar1_y + bar_height), (200, 200, 200), -1)
    cv.rectangle(output, (bar1_x, bar1_y), (bar1_x + cam_filled_width, bar1_y + bar_height), bar_color, -1)
    cv.rectangle(output, (bar1_x, bar1_y), (bar1_x + bar_width, bar1_y + bar_height), (0, 0, 0), 1)

    # Update previous frame and points
    if (leaf_next_pts is not None) and (len(leaf_next_pts) > 0):
        prev_leaf_pts = leaf_next_pts.reshape(-1, 1, 2)
    if (frame_next_pts is not None) and (len(frame_next_pts) > 0):
        frame_prev_pts = frame_next_pts.reshape(-1, 1, 2)

    #binmask
    if background_mask is not None:
        mask_color = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)
        mask_small = cv.resize(mask_color, (160, 120))
        x_offset = 20
        y_offset = output.shape[0] - 20 - mask_small.shape[0]
        output[y_offset:y_offset + mask_small.shape[0], x_offset:x_offset + mask_small.shape[1]] = mask_small

    out.write(output)
    
    vars["prev_gray"] = gray.copy()
    vars["frame_prev_pts"] = frame_next_pts.reshape(-1, 1, 2) if frame_next_pts.size > 0 else np.array([]).reshape(-1, 1, 2)
    vars["prev_leaf_pts"] = leaf_next_pts.reshape(-1, 1, 2) if leaf_next_pts.size > 0 else np.array([]).reshape(-1, 1, 2)
    vars["leaf_status"] = leaf_status
    vars["frame_pts_status"] = frame_pts_status
    vars["smoothed_motion"] = smoothed_motion
    vars["total_motion"] = total_motion
    vars["total_cam_motion"] = total_cam_motion
    vars["frame_cnt"] = frame_cnt
    vars["background_mask"] = background_mask
    vars["foreground_mask"] = foreground_mask
    vars["mask"] = mask

    return True
