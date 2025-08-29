import os
import cv2
import time
import json
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from main_mqtt import get_pose2D, get_pose3D  # Changed to relative import

app = FastAPI()

def format_elapsed_time(start_time):
    elapsed_seconds = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def action_time_counting(total_time):
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def process_frame(frame, output_dir):
    """Modified to return frame data without visualization"""
    get_time, json_2d = get_pose2D(frame, output_dir)
    pose_3d, json_3d = get_pose3D(frame, output_dir)
    
    # Create frame information
    return {
        "json_2d": json_2d,
        "json_3d": json_3d,
        "processing_time": time.time() - get_time if get_time else 0
    }

def process_video(video_path, gpu='0'):
    """Modified video processing function"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    results = []
    
    # Create output directory relative to current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "myoutput", "live")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_counter = 0
    start_time = time.time()
    action_time_list = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % 30 == 0:
                frame_data = process_frame(frame, output_dir)
                frame_data["frame_name"] = f"frame_{frame_counter:06d}"
                
                # Add timing information
                frame_data["elapsed_time"] = format_elapsed_time(start_time)
                
                results.append(frame_data)

    finally:
        cap.release()

    return results

@app.post("/process-video/")
async def process_video_endpoint(
    gpu: str = '0',
    file: UploadFile = File(..., description="Video file to process")
):
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            video_path = tmp_file.name

        # Process video
        results = process_video(video_path, gpu)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    finally:
        if 'tmp_file' in locals():
            os.unlink(video_path)

    return JSONResponse(content={"frames": results})