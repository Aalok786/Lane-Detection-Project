import cv2
import numpy as np
from timeit import default_timer as timer

Lane_width = 3.7
input_scale = 4
outpt_scale = 4
threshold = 0.85
video = cv2.VideoCapture("input.mp4")

codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps =int(video.get(cv2.CAP_PROP_FPS))
video_width,video_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('results.avi', codec, video_fps, (video_width, video_height))


def Processing(frame):
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(frame, lower_white, upper_white)
    hls_result0 = cv2.bitwise_and(frame, frame, mask=mask)
    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([40, 255, 255])
    mask1 = cv2.inRange(frame, lower_yellow, upper_yellow)
    mask2 = cv2.bitwise_or(mask, mask1)
    hls_result1 = cv2.bitwise_and(frame, frame, mask=mask2)
    hls_result = cv2.bitwise_or(hls_result0, hls_result1)
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)
    return canny, thresh


def Perspective_Transformation(binary_img):
    # width,height  = (binary_img.shape[1], binary_img.shape[0])
    img_size = (frame1.shape[1], frame1.shape[0])
    width, height = img_size
    dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    x1 = width * 0.35
    y1 = height * 0.5
    x2 = width * 0.6
    y2 = height * 0.5
    print(x1, y1, x2, y2)
    src = np.float32([[x1, y1], [x2, y2], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    birdseye = cv2.warpPerspective(binary_img, matrix, (width, height))
    minv = cv2.getPerspectiveTransform(dst, src)
    height, width = birdseye.shape[:2]
    birdseyeLeft = birdseye[0:height, 0:width // 2]
    birdseyeRight = birdseye[0:height, width // 2:width]
    return birdseye, minv


def window_search(binary_warped):
    hist = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    result = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    mid = np.int64(hist.shape[0] / 2)
    leftlane = np.argmax(hist[:mid])
    rightlane = np.argmax(hist[mid:]) + mid
    slidingwindows = 20
    window_height = np.int64(binary_warped.shape[0] / slidingwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    current_left = leftlane
    current_right = rightlane
    margin = 70
    minpix = 20
    correct_left = []
    correct_right = []
    for window in range(slidingwindows):
        y_low = binary_warped.shape[0] - (window + 1) * window_height
        y_high = binary_warped.shape[0] - window * window_height
        w_xleft_low = current_left - margin
        w_xleft_high = current_left + margin
        w_xright_low = current_right - margin
        w_xright_high = current_right + margin
        cv2.rectangle(result, (w_xleft_low, y_low), (w_xleft_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(result, (w_xright_low, y_low), (w_xright_high, y_high), (0, 255, 0), 2)
        goodleftlane_pos = \
        ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= w_xleft_low) & (nonzerox < w_xleft_high)).nonzero()[0]
        goodrightlane_pos = \
        ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= w_xright_low) & (nonzerox < w_xright_high)).nonzero()[
            0]
        correct_left.append(goodleftlane_pos)
        correct_right.append(goodrightlane_pos)
        if len(goodleftlane_pos) > minpix:
            current_left = np.int64(np.mean(nonzerox[goodleftlane_pos]))
        if len(goodrightlane_pos) > minpix:
            current_right = np.int64(np.mean(nonzerox[goodrightlane_pos]))
    correct_left = np.concatenate(correct_left)
    correct_right = np.concatenate(correct_right)
    leftx = nonzerox[correct_left]
    lefty = nonzeroy[correct_left]
    rightx = nonzerox[correct_right]
    righty = nonzeroy[correct_right]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    line = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    fit_left = left_fit[0] * line ** 2 + left_fit[1] * line + left_fit[2]
    fit_right = right_fit[0] * line ** 2 + right_fit[1] * line + right_fit[2]
    result[nonzeroy[correct_left], nonzerox[correct_left]] = [255, 255, 255]
    result[nonzeroy[correct_right], nonzerox[correct_right]] = [255, 255, 255]
    right = np.asarray(tuple(zip(fit_right, line)), np.int32)
    left = np.asarray(tuple(zip(fit_left, line)), np.int32)
    cv2.polylines(result, [right], False, (1, 1, 0), thickness=5)
    cv2.polylines(result, [left], False, (1, 1, 0), thickness=5)
    return line, mid, leftlane, rightlane, left_fit, right_fit, fit_left, fit_right, result, leftx, rightx


def radius_of_curvature(line, fit_left, fit_right):
    max_y = np.max(line)
    left_fit_cr = np.polyfit(line * ym_per_pix, fit_left * xm_per_pix, 2)
    right_fit_cr = np.polyfit(line * ym_per_pix, fit_right * xm_per_pix, 2)
    left_curved_rad = ((1 + (2 * left_fit_cr[0] * max_y * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curved_rad = ((1 + (2 * right_fit_cr[0] * max_y * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    if fit_left[0] - fit_left[-1] > 60:
        curve_direction = 'Right Curve'
    elif fit_left[-1] - fit_left[0] > 60:
        curve_direction = 'Left Curve'
    else:
        curve_direction = 'Straight'
    curve_rad = (left_curved_rad + right_curved_rad) / 2.0
    return curve_rad, curve_direction


def draw_lane_lines(original_image, warped_image, letfx, rightx, fitx_left, fitx_right, ploty, offcenter):
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([fitx_left, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fitx_right, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    mean_x = np.mean((fitx_left, fitx_right), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
    if round(offcenter) > threshold:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))
    newwarp = cv2.warpPerspective(color_warp, minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return pts_mean, result


def offCenter(mid, left, right):
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # driving right off
        offset = a / frame_width * Lane_width - Lane_width / 2.0
    else:  # driving left off
        offset = Lane_width / 2.0 - b / frame_width * Lane_width
    # deviation = offset * xm_per_pix
    direction = "Right" if offset < 0 else "Left"
    return offset, direction


def compute_car_offcenter(ploty, left_fitx, right_fitx, undist):
    # Create an image to draw the lines on
    height = undist.shape[0]
    width = undist.shape[1]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_l = left_fitx[height - 1]
    bottom_r = right_fitx[0]

    offcenter = off_center(bottom_l, width / 2.0, bottom_r)

    return offcenter, pts


def addText(img, radius, direction, deviation, devDirection):
    # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_TRIPLEX
    bg = cv2.imread("1.jpeg")
    #cv2.rectangle(img, (30, 30), (650, 300), (0, 0, 0), -255)
    if (direction == 'Straight'):
        text1 = 'Stay Straight'
    elif (direction == 'Left Curve' and deviation < 2):
        text1 = 'Slightly Turn Left '
    elif (direction == 'Right Curve' and deviation < 2):
        text1 = 'Slightly Turn Right '
    elif (direction == 'Left Curve' and deviation > 2):
        text1 = 'Turn Left ahead'
    elif (direction == 'Right Curve' and deviation > 2):
        text1 = 'Turn Right ahead '
    else:
        text1 = str(direction)
    if round(deviation) > threshold:
        cv2.putText(img, 'Stay in Lane', (260, 100), font, 0.8, (0, 0, 225), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'Good Lane Keeping', (260, 100), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    fps_text = str(round(fps)) + 'fps'
    # cv2.imshow('Frame',left)
    cv2.putText(img, 'LANE STATUS : ', (50, 100), font, 0.8, (252, 3, 144), 2)
    cv2.putText(img, 'DRIVER ASSISTANCE :', (50, 150), font, 0.8, (252, 3, 144), 2)
    cv2.putText(img, text1, (350, 150), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'Frame_per_second :', (50, 250), font, 0.8, (252, 3, 144), 2)
    cv2.putText(img, fps_text, (350, 250), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "OFFCENTRE : ", (50, 200), font, 0.8, (252, 3, 144), 2)
    deviation_text = str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
    cv2.putText(img, deviation_text, (230, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    return img,bg


while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture("input.mp4")
        continue
    start = timer()
    frame1 = cv2.resize(frame, (950, 700), interpolation=cv2.INTER_AREA)
    img_size = (frame1.shape[1], frame1.shape[0])
    width, height = img_size
    frame_width, frame_height = (frame.shape[1], frame.shape[0])
    canny, thresh = Processing(frame1)
    birdeye, minv = Perspective_Transformation(thresh)
    line, mid, leftlane, rightlane, left_fit, right_fit, fit_left, fit_right, output, leftx, rightx = window_search(
        birdeye)
    ym_per_pix = 30 / (frame_height / input_scale)
    xm_per_pix = 3.7 / (700 / input_scale)
    curve_radius, direction_1 = radius_of_curvature(line, fit_left, fit_right)
    deviation, direction_2 = offCenter(mid, leftlane, rightlane)
    mean, result = draw_lane_lines(frame1, thresh, leftx, rightx, fit_left, fit_right, line, deviation)
    end = timer()
    # print(deviation)

    fps = 1 / (end - start)

    final = addText(result, curve_radius, direction_1, deviation, direction_2)
    cv2.imshow("Frame1", result)
    out.write(result)


    if cv2.waitKey(1) == ord('q'):
        break
video.release()
out.release()
cv2.destroyAllWindows()