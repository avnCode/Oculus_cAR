import cv2
import numpy as np
from matplotlib import pyplot as plt

def prepocessing_img(frame):
    frame=cv2.putText(frame,'Oculus cAR',(30,30), font, 0.6,(255,0,255),2,cv2.LINE_AA)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    _, sxbinary = cv2.threshold(hls[:, :, 1], 120, 255, cv2.THRESH_BINARY)

    # Canny Edge detection
    canny=cv2.Canny(sxbinary, threshold1=120, threshold2=255)
    sxbinary=cv2.bitwise_not(canny)

    _, s_binary = cv2.threshold(hls[:, :, 2], 80, 255, cv2.THRESH_BINARY)
    _, r_thresh = cv2.threshold(frame[:, :, 1], 80, 255, cv2.THRESH_BINARY)
    rs_binary = cv2.bitwise_and(s_binary, r_thresh)
    return rs_binary,sxbinary,frame

def prespective_transform(rs_binary):
    _to_bird_eye_matrix = cv2.getPerspectiveTransform(roi_points, desired_roi_points)
    _bird_eye_frame = cv2.warpPerspective(rs_binary, _to_bird_eye_matrix, (_width, _height))
    (thresh, binary_warped) = cv2.threshold(_bird_eye_frame, 127, 255, cv2.THRESH_BINARY)
    binary_copy = binary_warped.copy()
    warped_plot = cv2.polylines(binary_copy, np.int32([desired_roi_points]), True, (147, 20, 255), 3)

    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    return binary_warped,histogram

def curve_fitting(warped_frame,histogram,reSized_frame):
    mid_way = int(histogram.shape[0] / 2)
    left_way = np.argmax(histogram[:mid_way])
    right_way = np.argmax(histogram[mid_way:]) + mid_way

    # Find the x and y coordinates of all the nonzero (i.e. white) pixels in the frame.
    nonzero = warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Store the pixel indices for the left and right lane lines
    left_lane_inds = []
    right_lane_inds = []

    leftx_current = left_way
    rightx_current = right_way
    frame_sliding_window = warped_frame.copy()

    # Sliding window parameters
    no_of_windows = 10
    margin = int((1 / 12) * _width)  # Window width is +/- margin
    minpix = int((1 / 24) * _width)  # Min no. of pixels to recenter window

    # Set the height of the sliding windows
    window_height = int(warped_frame.shape[0] / no_of_windows)

    for window in range(no_of_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_frame.shape[0] - (window + 1) * window_height
        win_y_high = warped_frame.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (
            win_xleft_high, win_y_high), (255, 255, 255), 2)
        cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (
            win_xright_high, win_y_high), (255, 255, 255), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    try:
        # Fit a second order polynomial curve to the pixel coordinates for the left and right lane lines
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Create the x and y values to plot on the image
        ploty = np.linspace(0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except Exception as e:
        print(f"Error in curve fitting: {e}")
    # Generate an image to visualize the result [BGR}
    out_img = np.dstack((frame_sliding_window, frame_sliding_window, frame_sliding_window)) * 255

    # Add color to the left line pixels and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate an image to draw the lane lines on
    warp_zero = np.zeros_like(warped_frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    try:
        pts_left = np.array([np.transpose(np.vstack([
            left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
            right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.polylines(color_warp, np.int_([pts]),False, (255, 0, 255 ))
    except Exception as e:
        print(f"Exception was due to polyfilt  reflected here: {e}")

    _to_car_eye_matrix = cv2.getPerspectiveTransform(desired_roi_points, roi_points)
    _car_eye_frame = cv2.warpPerspective(color_warp, _to_car_eye_matrix, (_width, _height))
    # blended = cv2.addWeighted(frame, 0.5, _car_eye_frame, 1, 0)

    # frame_copy = frame.copy()
    # h,w=_car_eye_frame.shape[:2]

    blended = cv2.addWeighted(reSized_frame,1, _car_eye_frame,1,0)
    return blended

def plotter(input,output):
    # Plotting
    figure, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(input)
    ax1.set_title("Frame")
    ax2.imshow(output)
    ax2.set_title("car_Eye")
    plt.show()

def video_Lane_detector(filePath:str):
    outVid_name = f"{filePath}.avi"
    cap = cv2.VideoCapture(filePath)
    cod = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outVid_name, cod, 20.0, (360, 360))
    while (cap.isOpened()):
        _, frame = cap.read()

        if frame is None:
            cap.release()
            cv2.destroyAllWindows()
            print("Frame ended!!!")
            break

        frame = cv2.resize(frame, size)
        preprocessed_image, preprocessed_image1,resized_frame = prepocessing_img(frame)
        warped_frame, hist = prespective_transform(preprocessed_image)
        img = curve_fitting(warped_frame, hist,resized_frame)
        out.write(img)
        cv2.imshow("result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def image_Lane_detector(filename):
    frame1 = cv2.imread(filename)
    frame1 = cv2.resize(frame1, size)
    preprocessed_image,preprocessed_image1,r=prepocessing_img(frame1)
    warped_frame,hist= prespective_transform(preprocessed_image)
    img=curve_fitting(warped_frame,hist,frame1)
    plotter(frame1,img)



if __name__ == '__main__':
    size = (350,350)
    _height, _width = size
    font = cv2.FONT_HERSHEY_SIMPLEX

    padding = int(0.2 * _width)  # padding from side of the image in pixels
    desired_roi_points = np.float32([
        [padding, 0],  # Top-left corner
        [padding, _height],  # Bottom-left corner
        [_width - padding, _height],  # Bottom-right corner
        [_width - padding, 0]  # Top-right corner
    ])

    roi_points = np.float32([
        (160, 230),  # Top-left corner
        (30, 350),  # Bottom-left corner
        (320, 350),  # Bottom-right corner
        (240, 230)  # Top-right corner
    ])

    # roi_points = np.float32([
    #     (185, 170),  # Top-left corner
    #     (90, 285),  # Bottom-left corner
    #     (320, 285),  # Bottom-right corner
    #     (265, 170)  # Top-right corner
    # ])

    # video_Lane_detector("test1.mp4")
    # video_Lane_detector("test2.mp4")
    # video_Lane_detector("test_gdrive2.mp4")
    # video_Lane_detector("test_gdrive.mkv")
    # video_Lane_detector("test3.mp4")
    video_Lane_detector("test4.avi")

    # video_Lane_detector("test5.mp4")
    # video_Lane_detector("project_video.mp4")

    # for i in range (1,7):
    #     image=f"img{i}.jpg"
    #     image_Lane_detector(image)
