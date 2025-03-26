import os
import glob
import cv2

def process_video(class_label, video_path, output_dir, clip_length=96, stride=48):
    """
    Extract overlapping 96-frame clips from a video with a given stride,
    pad leftover frames if needed, and save each clip as an .avi file.
    """
    # Create the class-specific output folder if it doesn't exist
    class_output_dir = os.path.join(output_dir, class_label)
    os.makedirs(class_output_dir, exist_ok=True)

    # Open the video and extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Attempt to retrieve the video's FPS; default to 30 if unavailable
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # No resizing is done per your request
        frames.append(frame)
    
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        print(f"Warning: {video_path} has no frames or could not be opened.")
        return

    # Use the original video filename (without extension) as part of the clip filename
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    
    clip_count = 1
    start_index = 0

    # Process overlapping clips
    while start_index < total_frames:
        end_index = start_index + clip_length
        clip_frames = frames[start_index:end_index]

        # If the final clip has fewer frames than required, pad with the last frame
        if (len(clip_frames) > 48) and (len(clip_frames) < clip_length):
            last_frame = clip_frames[-1]
            while len(clip_frames) < clip_length:
                clip_frames.append(last_frame)
        
        # Include the original video's name in the output filename to avoid overwrites
        output_filename = f"{class_label}_{video_base}_clip_{clip_count:02d}.avi"
        clip_path = os.path.join(class_output_dir, output_filename)
        save_clip_as_avi(clip_frames, clip_path, fps)
        print(f"Saved {clip_path} [{len(clip_frames)} frames]")
        
        clip_count += 1
        start_index += stride

def save_clip_as_avi(frames, output_path, fps=30.0):
    """
    Save a list of frames as an .avi video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def main():
    # Define your input and output directories
    data_dir = "data/video data"               # Folder containing class subfolders with videos
    output_dir = "data/processed video clips"     # Folder to store the output clips

    clip_length = 96
    stride = 48

    # Iterate through each class folder in the data directory
    for class_label in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, class_label)
        if os.path.isdir(class_folder):
            # Get all .mp4 files in this class folder
            video_files = glob.glob(os.path.join(class_folder, "*.mp4"))
            for video_path in video_files:
                print(f"Processing {video_path} for class '{class_label}'...")
                process_video(class_label, video_path, output_dir, clip_length, stride)
                
    print("All videos processed successfully.")

if __name__ == "__main__":
    main()
