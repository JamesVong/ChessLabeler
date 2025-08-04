import cv2
import os


class VideoUtils():
    def __init__(self, directory="DownloadedVideos"):
        self.video_directory = directory
    
    def getVideoFrames(self, video_name, interval_seconds=50):
        captured_frames = []        
        
        for video_file in os.listdir(self.video_directory):
            video_path = os.path.join(self.video_directory, video_file)
            
            if video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')) and video_name in video_path:
                video_capture = cv2.VideoCapture(video_path)
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps * interval_seconds)
                
                frame_count = 0
                while True:
                    # Set position to the next frame to capture
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    success, frame = video_capture.read()
                    
                    if success:
                        captured_frames.append((int(frame_count / fps), frame))
                        if len(captured_frames) % 100 == 0:
                            print(f"Captured frame at {frame_count / fps:.2f} seconds in {video_file}")
                        frame_count += frame_interval
                    else:
                        break
        
                video_capture.release()
                
        return captured_frames