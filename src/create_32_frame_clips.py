import os
import glob
import cv2

def split_video_into_clips(video_path, output_dir, clip_length=32):
    """
    Split a 96-frame video into non-overlapping clips of 32 frames each.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    total_frames = len(frames)

    if total_frames != 96:
        print(f"Warning: {video_path} has {total_frames} frames (expected 96). Skipping.")
        return

    num_clips = total_frames // clip_length

    for i in range(num_clips):
        start = i * clip_length
        end = start + clip_length
        clip_frames = frames[start:end]
        
        output_filename = f"{video_name}_clip_{i+1:02d}.avi"
        output_path = os.path.join(output_dir, output_filename)
        save_clip_as_avi(clip_frames, output_path, fps)
        print(f"Saved {output_path} [{len(clip_frames)} frames]")

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
    input_dir = r"C:\Users\jashw\Desktop\Video Surveillance\data\processed video clips"        # Replace with your actual input folder path
    output_dir = r"D:\Processed_video_clips"        # Replace with your actual output folder path

    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))

    for video_path in video_files:
        print(f"Processing: {video_path}")
        split_video_into_clips(video_path, output_dir)

    print("All videos processed successfully.")

if __name__ == "__main__":
    main()
