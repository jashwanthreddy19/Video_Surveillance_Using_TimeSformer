# backend/main.py
import os
import subprocess
import cv2
import numpy as np
import asyncio
import shutil
import math
import sys
import torch
from torchvision import transforms
sys.path.append(r"C:\Users\jashw\Desktop\Video Surveillance")
from models.TimeSformer.timesformer.models.vit import TimeSformer

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Configuration ---
FRAMES_PER_CHUNK = 96
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHUNK_DIR = os.path.join(UPLOAD_DIR, "chunks") # Chunks stored inside uploads
STATIC_DIR = os.path.join(BASE_DIR, "static") # For abnormal clips
MODEL_PATH = r"C:\Users\jashw\Desktop\Video Surveillance\models\data\trained_models\checkpoints\best_model_epoch11_acc90.34.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class_list = ['Abuse', 'Explosion', 'Fighting', 'RoadAccident', 'Robbery', 'Shooting', 'Vandalism', 'Normal Video']

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static file serving
# IMPORTANT: Serve uploads so frontend can get display video
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
# Serve static for abnormal clips
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Loading the model and moving to GPU
print(f"Loading fine-tuned model from: {MODEL_PATH}")

    # Create the TimeSformer model instance without pretrained_model
model = TimeSformer(img_size=224, num_classes=8, num_frames=96,
                    attention_type='divided_space_time', pretrained_model=None) #pretrained_model=None

# Load the state_dict from your fine-tuned model
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint, strict=True) #strict true
model.to(DEVICE)

print("Model loaded successfully!")


# --- Utility Functions ---

def clean_uploads_directory():
    """Cleans directories for a fresh upload session."""
    # Be careful with rmtree in production!
    for dir_path in [CHUNK_DIR, STATIC_DIR]: # Keep UPLOAD_DIR base, clean subdirs
       if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
       os.makedirs(dir_path, exist_ok=True)
    # Clean loose files in UPLOAD_DIR except subdirs
    for item in os.listdir(UPLOAD_DIR):
        item_path = os.path.join(UPLOAD_DIR, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
    print("Cleaned CHUNK_DIR and STATIC_DIR.")


def run_ffmpeg(cmd_args):
    """Runs an FFmpeg command."""
    print("Running command:", " ".join(cmd_args))
    try:
        result = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        print("FFmpeg STDOUT:", result.stdout)
        print("FFmpeg STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error (Exit Code {e.returncode}):")
        print("FFmpeg STDOUT:", e.stdout)
        print("FFmpeg STDERR:", e.stderr)
        raise # Re-raise the exception to signal failure


def convert_video(input_path: str, output_path: str):
    """Converts video using FFmpeg."""
    cmd = ["ffmpeg", "-y", "-i", input_path, output_path]
    run_ffmpeg(cmd)
    print(f"Converted {input_path} to {output_path}")

# --- Chunk Splitting ---

def split_video_into_chunks(video_path: str, chunk_dir: str, frames_per_chunk: int = FRAMES_PER_CHUNK):
    """Splits video into chunks and returns total chunks and fps."""
    os.makedirs(chunk_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not determine FPS, defaulting to 30.")
        fps = 30.0 # Default FPS if detection fails

    chunk_index = 0
    total_frames_processed = 0

    while True:
        frames = []
        for _ in range(frames_per_chunk):
            ret, frame = cap.read()
            if not ret:
                break # End of video
            frames.append(frame)

        # Only save if we got a full chunk's worth of frames
        if len(frames) == frames_per_chunk:
            chunk_filename = f"chunk_{chunk_index:04d}.avi" # Use 4 digits for more chunks
            chunk_path = os.path.join(chunk_dir, chunk_filename)

            # Get frame dimensions from the first frame of the chunk
            if frames:
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID') # AVI codec
                # Use detected FPS
                out = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))
                for frame in frames:
                    out.write(frame)
                out.release()
                # print(f"Saved chunk: {chunk_filename}")
                total_frames_processed += len(frames)
                chunk_index += 1
            else:
                 # Should not happen if len(frames) == frames_per_chunk, but safety check
                break

        else:
            # Reached end of video, partial chunk discarded
            print(f"End of video reached. Discarding last partial chunk with {len(frames)} frames.")
            break

    cap.release()
    print(f"Split video into {chunk_index} chunks of {frames_per_chunk} frames each at {fps:.2f} FPS.")
    return chunk_index, fps # Return total number of chunks created and fps

# --- Model Processing (Dummy) ---

def read_video(video_path):
        """Reads video and returns frames as numpy array."""
        clip_len = 96
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()

        # If the video has fewer frames than clip_len, pad with the last frame
        if len(frames) < clip_len:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            while len(frames) < clip_len:
                frames.append(last_frame)
        
        return np.array(frames)


def process_chunk(chunk_path: str, model,class_list):
    """Dummy function to process a single chunk."""
    # In a real scenario, load your model and process the video chunk
    # This dummy version uses brightness as a placeholder
    try:
        frames = read_video(chunk_path)
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if transform:
                frames = [transform(frame) for frame in frames]  # Each frame becomes [C, H, W]
            
        frames = torch.stack(frames)  # Shape: (T, C, H, W)
        frames_tensor = frames.permute(1, 0, 2, 3).float()
        frames_tensor = frames_tensor.unsqueeze(0)
        frames_tensor = frames_tensor.to(DEVICE)
        with torch.no_grad():
            outputs = model(frames_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        predicted = class_list[predicted]
        print(f"The Predicted Class for chunk is : {predicted}")
        return predicted
    except Exception as e:
        print(f"Error processing chunk {chunk_path}: {e}")
        return "error"

def convert_chunk_to_mp4(avi_chunk_path: str, output_dir: str):
    """Converts an AVI chunk to MP4 for alerts."""
    base_filename = os.path.basename(avi_chunk_path)
    mp4_filename = f"alert_{base_filename[:-4]}.mp4"
    output_path = os.path.join(output_dir, mp4_filename)
    convert_video(avi_chunk_path, output_path)
    # Return the relative path for the URL
    return f"/static/{mp4_filename}" # Path relative to static mount

# --- API Endpoints ---

@app.post("/api/upload")
async def upload_and_preprocess_video(file: UploadFile = File(...)):
    """Handles video upload, conversion, chunking, and returns info."""
    
    # Clean directories for the new session
    clean_uploads_directory() 

    filename = file.filename
    if not filename:
         return JSONResponse({"error": "Filename cannot be empty"}, status_code=400)
    
    file_ext = os.path.splitext(filename)[-1].lower()
    base_name = os.path.splitext(filename)[0] # Get name without extension

    upload_path = os.path.join(UPLOAD_DIR, filename)
    processing_video_path = None # Path to the AVI file we will process
    display_path_relative = None # Path relative to /uploads mount point

    try:
        # Save the uploaded file
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"Uploaded file saved to {upload_path}")

        # Convert and set paths
        if file_ext == ".avi":
            # Convert AVI to MP4 for display
            display_path_abs = os.path.join(UPLOAD_DIR, f"{base_name}_display.mp4")
            convert_video(upload_path, display_path_abs)
            display_path_relative = f"/uploads/{os.path.basename(display_path_abs)}"
            processing_video_path = upload_path # Process the original AVI
        elif file_ext == ".mp4":
            # Convert MP4 to AVI for processing
            processing_video_path = os.path.join(UPLOAD_DIR, f"{base_name}_processing.avi")
            convert_video(upload_path, processing_video_path)
            display_path_relative = f"/uploads/{filename}" # Display the original MP4
        else:
            os.remove(upload_path) # Clean up unsupported upload
            return JSONResponse({"error": "Unsupported file type. Please upload MP4 or AVI."}, status_code=400)

        # Split the processing AVI into chunks
        if not processing_video_path or not os.path.exists(processing_video_path):
             raise ValueError("Processing video path is not valid.")

        total_chunks_created, fps = split_video_into_chunks(processing_video_path, CHUNK_DIR)

        if total_chunks_created == 0:
             raise ValueError("Video processing failed: No chunks were created. Check video format or length.")


        return JSONResponse({
            "message": "Upload and pre-processing complete.",
            "displayUrl": display_path_relative, # URL relative to server root
            "totalChunks": total_chunks_created,
            "fps": fps,
        })

    except FileNotFoundError as e:
         print(f"Error: File not found during processing - {e}")
         return JSONResponse({"error": f"File operation error: {e}"}, status_code=500)
    except ValueError as e:
         print(f"Error: {e}")
         return JSONResponse({"error": str(e)}, status_code=500)
    except subprocess.CalledProcessError as e:
         print(f"Error: FFmpeg conversion failed - {e}")
         return JSONResponse({"error": f"Video conversion failed. Ensure FFmpeg is installed and the video format is supported."}, status_code=500)
    except Exception as e:
        # Catch other potential errors during processing
        print(f"An unexpected error occurred: {e}")
        # Clean up potentially corrupted files if needed
        # clean_uploads_directory() # Optional: Decide if cleanup is needed on failure
        return JSONResponse({"error": f"An unexpected error occurred during processing: {e}"}, status_code=500)


# --- WebSocket Endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket communication for chunk processing requests."""
    await websocket.accept()
    print("WebSocket connection accepted.")
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "processChunk":
                # Expect 0-based index from frontend
                chunk_index = data.get("chunkIndex")

                if chunk_index is None or not isinstance(chunk_index, int) or chunk_index < 0:
                     print(f"Invalid chunk index received: {chunk_index}")
                     continue # Ignore invalid messages

                # Construct chunk path using 0-based index
                chunk_filename = f"chunk_{chunk_index:04d}.avi"
                chunk_path = os.path.join(CHUNK_DIR, chunk_filename)

                print(f"Received request for chunk index: {chunk_index} (Path: {chunk_path})")

                if not os.path.exists(chunk_path):
                    print(f"Chunk file does not exist: {chunk_path}")
                    # Optionally send an error back to frontend
                    # await websocket.send_json({"error": "Chunk not found", "sourceChunkIndex": chunk_index})
                    continue

                # Process the chunk
                result = process_chunk(chunk_path,model,class_list)
                alert_data = {
                     "abnormal": False,
                     "label": "Normal",
                     "clipUrl": None,
                     "sourceChunkIndex": chunk_index # Echo back the index
                 }

                if result == "abnormal":
                    print(f"Abnormality detected in chunk {chunk_index}")
                    try:
                        clip_url = convert_chunk_to_mp4(chunk_path, STATIC_DIR)
                        alert_data.update({
                            "abnormal": True,
                            "label": "Abnormal Event Detected", # Replace with actual model label
                            "clipUrl": clip_url
                        })
                    except Exception as e:
                         print(f"Failed to convert abnormal chunk {chunk_index} to MP4: {e}")
                         alert_data["label"] = "Processing Error" # Indicate error instead

                elif result == "error":
                     print(f"Error processing chunk {chunk_index}")
                     alert_data["label"] = "Processing Error"


                # Send result back to frontend
                await websocket.send_json(alert_data)
            else:
                print(f"Received unknown message type: {message_type}")

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        # Ensure connection is closed gracefully if possible
        # await websocket.close(code=1011) # Internal Server Error code

# --- Run Application ---

if __name__ == "__main__":
    import uvicorn
    # Consider adding reload=True for development
    uvicorn.run(app, host="0.0.0.0", port=8000)