# after scene rendering, generate the gif and mesh visualization for each scene
import os, sys
# current_dir=os.path.abspath(__file__)
# sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "utils")))

import argparse

from utils.gif_video_generate import *
from utils.visualize_mesh import *
import glob


def list_renderable(dataset, output_dir, subset):
    pattern = f"./{output_dir}/{dataset}/{subset}/*/final/train/ours_20000/meshes/part_0.ply"
    scenes = []
    for p in sorted(glob.glob(pattern)):
        scene = p.split(f"/{subset}/")[1].split("/")[0]
        scenes.append(scene)
    return scenes



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="videoartgs", help="dataset name")
    parser.add_argument("--subset", type=str, default="sapien", help="subset name")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output directory")
    
    args = parser.parse_args()
    dataset = args.dataset
    subset = args.subset
    output_dir = args.output_dir
    
    VideoArtGS_scenes = list_renderable(dataset, output_dir, subset)
    for scene in VideoArtGS_scenes:
        generate_video_gif(dataset=dataset, subset=subset, scene=scene, output_dir=output_dir)
        color_parts(scene, dataset="videoartgs", output_dir=output_dir, subset=subset, base=".")