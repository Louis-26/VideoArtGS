"""
Tool script to generate .mp4 and .gif from a number of images
"""
import os
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import glob
import subprocess

def video2gif_ffmpeg(video_file):
    gif_file = video_file.replace('.mkv', '.gif').replace('.mp4', '.gif')
    os.makedirs(os.path.dirname(gif_file), exist_ok=True)
    # Ensure -y is placed before the output file, or right at the end
    cmd = f'ffmpeg -i {video_file} -vf "fps=10,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -y {gif_file}'
    subprocess.run(cmd, shell=True, check=True)

def video2gif(video_file):
    gif_file = video_file.replace('.mp4', '.gif')
    clip = VideoFileClip(video_file)
    clip.write_gif(gif_file)
    clip.close()

def generate_video_ffmpeg(img_path, video_name, fps=30):
    if os.path.exists(video_name):
        # Use python's native remove instead of calling 'rm' via shell
        os.remove(video_name)
        
    # 🌟 FIX 1: We securely place -fps_mode 0 AFTER the input %06d.png
    cmd = f'ffmpeg -y -framerate {fps} -i {img_path}/%06d.png -c:v libx264 -crf 0 {video_name}'
    subprocess.run(cmd, shell=True, check=True)

def generate_video(img_path, video_name, fps=15, brighten=False, white_background=False):
    # imgs: list of img tensors [3, H, W]
    imgs = sorted(glob.glob(f'{img_path}/*.png'))
    # imgs = imgs[100:]
    imgs = [cv2.imread(img, -1) for img in imgs] # rgba
    # transfer to rgb with white background
    print(f'Generating video {video_name} with {len(imgs)} frames')
    for i, img in enumerate(imgs):
        if img is None:
            print(f'img {i} is None')
            continue
        if img.shape[-1] == 4:
            imgs[i] = img[:, :, :3] * (img[:, :, 3:4] / 255.0) 
            if white_background:
                imgs[i] = imgs[i] + 255 * (1 - img[:, :, 3:4] / 255.0)
            imgs[i] = imgs[i].astype(np.uint8)
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for img in imgs:
        if brighten:
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=1)
        video.write(img)
    video.release()
    
    
# video and gif generation
def generate_video_gif(dataset, subset, scene, output_dir="outputs"):
    file_dir = f"./{output_dir}/{dataset}/{subset}/{scene}/final/train/ours_20000"
    img_path = f"{file_dir}/renders/-1"
    video_file = f'{file_dir}/{scene}_video.mp4'
    
    # video
    if not os.path.exists(video_file):
        generate_video_ffmpeg(img_path, video_file, 20)
        
    # gif
    if not os.path.exists(f'{file_dir}/{scene}_video.gif'):
        video2gif_ffmpeg(video_file)