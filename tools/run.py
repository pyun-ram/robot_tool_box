import fire
from pathlib import Path
from PIL import Image
import imageio
import numpy as np
import os
import re
from dataclasses import dataclass
from tqdm import tqdm
@dataclass
class Config:
    camera_names: ['cam0', 'cam1', 'cam2', 'cam3']
    camera_frequency: 20
    camera_resolution: (640, 480)
    camera_sync_precision: 8 # ms

class Runner:
    def organize_data(self, raw_episode_dir: str, episode_dir: str) -> None:
        # get start timestamp
        # get timestamp list
        # for each time stamp
        ## get closest cam0 data, and check its timestamp is within camera_sync_precision
        ## get closest cam1 data, and check its timestamp is within camera_sync_precision
        ## get closest cam2 data, and check its timestamp is within camera_sync_precision
        ## get closest cam3 data, and check its timestamp is within camera_sync_precision
        ## get closest gripper data, and check its timestamp is within camera_sync_precision
        ## save to save_episode_dir
        raise NotImplementedError("Not implemented")
    
    def check_calibration(self, episode_dir: str) -> None:
        # for each idx
        ## visualize gripper end-effector pose in cam0 pointcloud
        ## visualize gripper end-effector pose in cam1 pointcloud
        ## visualize gripper end-effector pose in cam2 pointcloud
        ## visualize gripper end-effector pose in cam3 pointcloud
        ## visualize merged pointcloud [cam0, cam1, cam2, cam3, gripper]
        raise NotImplementedError("Not implemented")
    
    def visualize_image(self, image_dir: str) -> None:
        image1_dir='/home/pyun/Docker/RM01/Data/episode0/cam1_color_image_raw_compressed'
        image2_dir='/home/pyun/Docker/RM01/Data/episode0/cam2_color_image_raw_compressed'
        image3_dir='/home/pyun/Docker/RM01/Data/episode0/cam3_color_image_raw_compressed'
        image4_dir='/home/pyun/Docker/RM01/Data/episode0/cam4_color_image_raw_compressed'
        
        # get image path list (image1_dir)
        image1_path = Path(image1_dir)
        image_files = sorted([f for f in image1_path.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
        
        if not image_files:
            raise ValueError(f"No image files found in {image1_dir}")
        
        # Extract timestamps from filenames (format: YYYYMMDD_HHMMSS_mmm.png -> extract HHMMSS.mmm)
        def extract_timestamp(filepath: Path) -> float:
            """Extract timestamp from filename. Format: 20251220_172659_411.png -> 172659.411"""
            filename = filepath.stem
            # Match pattern: YYYYMMDD_HHMMSS_mmm
            # Extract HHMMSS and mmm, combine as HHMMSS.mmm
            match = re.search(r'_(\d{6})_(\d{3})$', filename)
            if match:
                time_part = match.group(1)  # e.g., "172659"
                ms_part = match.group(2)    # e.g., "411"
                timestamp_str = f"{time_part}.{ms_part}"  # e.g., "172659.411"
                return float(timestamp_str)
            # Fallback to file modification time
            return os.path.getmtime(filepath)
        
        # Get timestamps for image1 using absolute time
        image1_timestamps = [extract_timestamp(f) for f in image_files]
        t_start = image1_timestamps[0]  # First frame absolute timestamp
        t_end = image1_timestamps[-1]   # Last frame absolute timestamp
        
        # get timestamp list with image1 as reference [use the first frame as t0 and compute future timestampes at 1 Hz][freq 5 Hz]
        # 5 Hz means 0.2 seconds per frame
        frame_interval = 1  # seconds
        target_timestamps = np.arange(t_start, t_end + frame_interval, frame_interval)
        
        # Process each camera directory separately
        def load_camera_images(camera_dir: str) -> tuple:
            """Load images and timestamps for a camera directory."""
            cam_path = Path(camera_dir)
            files = sorted([f for f in cam_path.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
            if not files:
                return [], []
            # Extract absolute timestamps
            timestamps = [extract_timestamp(f) for f in files]
            return files, timestamps
        
        # Load images for each camera
        cam1_files, cam1_timestamps = load_camera_images(image1_dir)
        cam2_files, cam2_timestamps = load_camera_images(image2_dir)
        cam3_files, cam3_timestamps = load_camera_images(image3_dir)
        cam4_files, cam4_timestamps = load_camera_images(image4_dir)
        
        def find_closest_image(target_time: float, files: list, timestamps: list) -> Path:
            """Find the image file with timestamp closest to target_time."""
            if not files or not timestamps:
                return None
            idx = np.argmin([abs(t - target_time) for t in timestamps])
            return files[idx]
        
        # for each timestamp
        frames = []
        for target_time in tqdm(target_timestamps):
            # Process each camera separately
            # Get closest image from cam1
            cam1_file = find_closest_image(target_time, cam1_files, cam1_timestamps)
            cam1_img = Image.open(cam1_file) if cam1_file and cam1_file.exists() else Image.new('RGB', (640, 480), color='black')
            
            # Get closest image from cam2
            cam2_file = find_closest_image(target_time, cam2_files, cam2_timestamps)
            cam2_img = Image.open(cam2_file) if cam2_file and cam2_file.exists() else Image.new('RGB', cam1_img.size, color='black')
            
            # Get closest image from cam3
            cam3_file = find_closest_image(target_time, cam3_files, cam3_timestamps)
            cam3_img = Image.open(cam3_file) if cam3_file and cam3_file.exists() else Image.new('RGB', cam1_img.size, color='black')
            
            # Get closest image from cam4
            cam4_file = find_closest_image(target_time, cam4_files, cam4_timestamps)
            cam4_img = Image.open(cam4_file) if cam4_file and cam4_file.exists() else Image.new('RGB', cam1_img.size, color='black')
            
            # Resize all images to same size if needed
            images = [cam1_img, cam2_img, cam3_img, cam4_img]
            sizes = [img.size for img in images]
            target_size = max(sizes, key=lambda x: x[0] * x[1])
            images = [img.resize(target_size) if img.size != target_size else img for img in images]
            
            # concatenate image [2x2] [cam1, cam2; cam3, cam4]
            img1, img2, img3, img4 = images
            top_row = np.hstack([np.array(img1), np.array(img2)])
            bottom_row = np.hstack([np.array(img3), np.array(img4)])
            combined = np.vstack([top_row, bottom_row])
            combined = Image.fromarray(combined).resize((640, 480))
            frames.append(combined)
        
        # visualize image with gif
        if frames:
            output_path = 'visualization.gif'
            imageio.mimsave(str(output_path), frames, duration=frame_interval, loop=0)
            print(f"GIF saved to {output_path}")
        else:
            raise ValueError("No frames to visualize")

if __name__ == "__main__":
    fire.Fire(Runner)
