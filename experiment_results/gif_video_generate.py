import os
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from glob import glob


def video2gif_ffmpeg(video_file):
    gif_file = video_file.replace('.mkv', '.gif').replace('.mp4', '.gif')
    os.makedirs(os.path.dirname(gif_file), exist_ok=True)
    cmd = f'ffmpeg -i {video_file} -vf "fps=10,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {gif_file} -y'
    os.system(cmd)


def video2gif(video_file):
    gif_file = video_file.replace('.mp4', '.gif')
    clip = VideoFileClip(video_file)
    clip.write_gif(gif_file)
    clip.close()

def generate_video_ffmpeg(img_path, video_name, fps=30):
    if os.path.exists(video_name):
        os.system(f'rm {video_name}')
    os.system(f'ffmpeg -framerate {fps} -vsync 0 -i {img_path}' + '/%06d.png -c:v libx264 -crf 0 ' + video_name)

def generate_video(img_path, video_name, fps=15, brighten=False, white_background=False):
    # imgs: list of img tensors [3, H, W]
    imgs = sorted(glob(f'{img_path}/*.png'))
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
def generate_video_gif(dataset, subset, scene):
    # dataset = 'videoartgs'
    # subset = 'sapien'
    # scene = '30666'
    file_dir = f"../outputs/{dataset}/{subset}/{scene}/final/train/ours_20000"
    img_path = f"{file_dir}/renders/-1"
    video_file = f'{file_dir}/{scene}_video.mp4'
    
    # video
    if not os.path.exists(video_file):
        # generate_video(img_path, video_file, 10)
        generate_video_ffmpeg(img_path, video_file, 20)
        # video2gif(video_file)
        
        
    # gif
    if not os.path.exists(f'{file_dir}/{scene}_video.gif'):
        video2gif_ffmpeg(video_file)