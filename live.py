import os
import cv2
import time
from main import get_pose2D, get_pose3D, merge_img


def format_elapsed_time(start_time):
    """Format elapsed time into HH:MM:SS format."""
    elapsed_seconds = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def action_time_counting(total_time):
    """Format action time into HH:MM:SS format."""
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def process_frame(frame, start_time, action_time_list):
    """Process each frame to compute pose and display necessary details."""
    output_dir = './demo/myoutput/live/'
    os.makedirs(output_dir, exist_ok=True)

    # Get 2D and 3D poses and merge images
    get_time = get_pose2D(frame, output_dir)
    pose_3d = get_pose3D(frame, output_dir)
    combined_img = merge_img(frame, output_dir)

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (100, 0, 255)  # Green color
    thickness = 2
    position_time = (frame.shape[1] - 140, 15)  # Top-right corner for elapsed time
    position_action_time = (frame.shape[1] - 140, 35)

    # Calculate and display action timer
    if get_time == 0:
        action_time_list.clear()
        cv2.putText(combined_img, "Action: 00:00:00", position_action_time, font, font_scale, font_color, thickness)
    else:
        human_action_timer = int(time.time() - get_time) + 6
        action_time_list.append(human_action_timer)
        action_timer = action_time_counting(sum(action_time_list))
        cv2.putText(combined_img, f"Action: {action_timer}", position_action_time, font, font_scale, font_color, thickness)

    # Display elapsed time
    elapsed_time_text = format_elapsed_time(start_time)
    cv2.putText(combined_img, f"Time: {elapsed_time_text}", position_time, font, font_scale, font_color, thickness)

    # Show images
    cv2.imshow("Original + 2D", combined_img)
    cv2.imshow("3D Pose", pose_3d)
    cv2.waitKey(10)


def capture_webcam_video():
    """Capture video from webcam and process frames."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    # Initialize video properties and timers
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_time = time.time()
    frame_counter = 0
    action_time_list = []

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame_counter += 1

        # Check if the current frame is the 300th frame
        if frame_counter % 200 == 0:
            print(f"Processing the {frame_counter}th frame.")
            process_frame(frame, start_time, action_time_list)

        # Exit on pressing 'Esc'
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC key code
            print("Exiting video capture...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (use 0 for CPU)')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Start video capture
    capture_webcam_video()
