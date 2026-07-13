import open3d as o3d
import numpy as np
import seaborn as sns
import glob, os

def color_parts(scene, dataset, output_dir="outputs", subset="sapien", base="."):
    train_dir = f"{base}/{output_dir}/{dataset}/{subset}/{scene}/final/train/ours_20000"
    mesh_dir = f"{train_dir}/meshes"
    parts = sorted(glob.glob(f"{mesh_dir}/part_*.ply"),
                   key=lambda p: int(p.split('part_')[1].split('.')[0]))
    n = len(parts)

    palette = np.array(sns.color_palette("hls", n))
    palette[0] = [0.5, 0.5, 0.5]   

    combined = o3d.geometry.TriangleMesh()
    for i, pf in enumerate(parts):
        m = o3d.io.read_triangle_mesh(pf)
        m.compute_vertex_normals()
        m.paint_uniform_color(palette[i])
        combined += m

    axes = sorted(glob.glob(f"{mesh_dir}/axis_*_p.ply"))
    axes = [a for a in axes if "_gt" not in a]
    for j, af in enumerate(axes):
        a = o3d.io.read_triangle_mesh(af)
        a.compute_vertex_normals()
        a.paint_uniform_color(palette[(j + 1) % n])
        combined += a

    out = f"{train_dir}/{scene}_colored.ply"
    o3d.io.write_triangle_mesh(out, combined)
    print(f"{scene}: {n} parts -> {out}")
    return out

def output_scene(scene_list):
    for scene in scene_list:
        try:
            color_parts(scene, dataset="videoartgs", base="..")
        except Exception as e:
            print(f"{scene}: {e}")