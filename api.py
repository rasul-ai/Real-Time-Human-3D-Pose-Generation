import os
import cv2
import time
import json
import threading
# import paho.mqtt.client as mqtt
from fastapi import FastAPI, UploadFile, File
import numpy as np

from demo.main_mqtt import get_pose2D, get_pose3D, merge_img

# # MQTT Settings
# BROKER = "192.168.1.100"  # Receiver's IP address
# PORT = 1883
# TOPIC = "pose/json"

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize MQTT Client
# mqtt_client = mqtt.Client()

# def connect_mqtt():
#     """Function to connect MQTT client in a separate thread."""
#     mqtt_client.connect(BROKER, PORT, 60)
#     mqtt_client.loop_start()  # Start the loop in a separate thread

# # Start MQTT client in background
# mqtt_thread = threading.Thread(target=connect_mqtt)
# mqtt_thread.start()

def process_frame(frame):
    """Process each frame to compute pose and return the JSON data."""
    output_dir = './demo/myoutput/live/'
    os.makedirs(output_dir, exist_ok=True)

    # Get 2D and 3D poses and merge images
    _, json_2d = get_pose2D(frame, output_dir)
    _, json_3d = get_pose3D(frame, output_dir)

    # Send via MQTT
    # if json_3d and json_2d:
    #     data_payload = {"2d_pose_json": json_2d, "3d_pose_json": json_3d}
    #     mqtt_client.publish(TOPIC, json.dumps(data_payload))
    #     print(f"Sent data via MQTT: {data_payload}")

    return json_2d, json_3d

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    """API endpoint to process video and return pose data."""
    video_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video file"}

    frame_counter = 0
    pose_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Process every 200th frame
        if frame_counter % 200 == 0:
            json_2d, json_3d = process_frame(frame)
            pose_results.append({"frame": frame_counter, "2d_pose_json": json_2d, "3d_pose_json": json_3d})

    cap.release()
    return {"poses": pose_results}

