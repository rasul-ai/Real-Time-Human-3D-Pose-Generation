import cv2
import os
import time
from main import get_pose2D, get_pose3D, merge_img


 
# Function to send 3D pose data via MQTT
def send_pose_3d_via_mqtt(pose_3d):
    """Send 3D pose data via MQTT in JSON format."""
    # Convert pose_3d (numpy array) to list of lists for easy JSON encoding
    pose_data = pose_3d.tolist()
    pose_json = json.dumps(pose_data)
    client.publish(topic, pose_json)  # Send the 3D pose via MQTT
    print("Sent 3D Pose Data via MQTT:", pose_json)
 
def process_frame(frame, start_time, action_time_list):
    """Process each frame to compute pose and display necessary details."""
    output_dir = './demo/myoutput/live/'
    os.makedirs(output_dir, exist_ok=True)
 
    # Get 2D and 3D poses and merge images
    get_time = get_pose2D(frame, output_dir)
    pose_3d = get_pose3D(frame, output_dir)
    combined_img = merge_img(frame, output_dir)
 
    # Send the 3D pose data via MQTT
    send_pose_3d_via_mqtt(pose_3d)  # Send pose_3d via MQTT
 
    # Define text properties for frame display
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (100, 0, 255)  # Green color
    thickness = 2
    position_time = (frame.shape[1] - 140, 15)  
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
 
 
def process_frame(frame, prev_time):
    # Track FPS
    curr_time = time.time()  # Get current time
    fps = 1 / (curr_time - prev_time)  # Calculate FPS
    prev_time = curr_time  # Update the previous time to the current time
 
    # Get pose information
    output_dir = './demo/myoutput/' + "live" + '/'
    os.makedirs(output_dir, exist_ok=True)
 
    # Get 2D poses from the frame
    poses_2d = get_pose2D(frame, output_dir)
 
    # Check if poses_2d is None, and set pose_count to 0 if so
    if poses_2d is None:85
        pose_count = 0
    else:
        pose_count = len(poses_2d)  # Count the detected poses
 
    # Get 3D pose
    pose_3d = get_pose3D(frame, output_dir)
    combined_img = merge_img(frame, output_dir)
 
    # Add FPS and pose count to the original + 2D combined image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_fps = f"FPS: {fps:.2f}"
    text_pose_count = f"Pose Count: {pose_count}"
 
    # Position for the text on the left side
    cv2.putText(combined_img, text_fps, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_img, text_pose_count, (10, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the images
    cv2.imshow("Original + 2D", combined_img)
    cv2.imshow("3D Pose", pose_3d)
    
    cv2.waitKey(10)  # Ensure that OpenCV is waiting for keypress
 
    return prev_time  # Return the updated time
 
 
def capture_webcam_video():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return None
 
    cv2.waitKey(5)
 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
 
    frame_counter = 0  # Initialize frame counter
    prev_time = time.time()  # Initialize the previous time for FPS calculation
 
    # Process frames in a loop
    while True:
        ret, frame = cap.read()  # Capture frame
        if not ret:
            print("Failed to capture frame")
            break
 
        frame_counter += 1  # Increment the frame counter
 
        # Process only every 30th frame
        if frame_counter % fps == 0:
            print(f"Processing frame {frame_counter}...")
 
            print("Press    ctrl + c     to exit.")
 
            # Process the captured frame
            prev_time = process_frame(frame, prev_time)
 
        # Press 'Esc' to stop the capture
        # key = cv2.waitKey(10) & 0xFF  # Check for keypress
        # if key == 27:  # ESC key code is 27
        #     print("Exiting video capture...")
        #     break
 
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    import argparse
    import os
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (use 0 for CPU)')
    args = parser.parse_args()
 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    capture_webcam_video()
 