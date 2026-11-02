#!/usr/bin/env python3
"""
utils/expand_ply.py

Parse and fully expand any PLY file into readable text containing all vertices,
all faces, and geometric metadata (centroid, bounding box).
Supports both binary (little-endian) and ASCII PLY formats.
Accepts relative paths from the repository root or absolute paths.
"""

import os
import sys
import struct
import argparse
from pathlib import Path


TYPE_MAP = {
    'char': ('b', 1), 'int8': ('b', 1),
    'uchar': ('B', 1), 'uint8': ('B', 1),
    'short': ('h', 2), 'int16': ('h', 2),
    'ushort': ('H', 2), 'uint16': ('H', 2),
    'int': ('i', 4), 'int32': ('i', 4),
    'uint': ('I', 4), 'uint32': ('I', 4),
    'float': ('f', 4), 'float32': ('f', 4),
    'double': ('d', 8), 'float64': ('d', 8)
}


def parse_ply(ply_path):
    """
    Parse a PLY file and extract all vertices, faces, and metadata.

    Args:
        ply_path (str or Path): Path to the PLY file.

    Returns:
        vertices (list of dict): Each dict contains 'id', 'x', 'y', 'z', and optional properties.
        faces (list of dict): Each dict contains 'id' and 'indices' (list of int vertex IDs).
        info (dict): Metadata including format, counts, centroid, and bounding box.
    """
    ply_path = Path(ply_path).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    with open(ply_path, 'rb') as f:
        # 1. Parse header
        ply_format = None
        num_verts = 0
        num_faces = 0
        vert_props = []
        face_list_type = None
        face_index_type = None

        current_element = None

        while True:
            line_bytes = f.readline()
            if not line_bytes:
                break
            line = line_bytes.decode('ascii', errors='ignore').strip()
            if not line or line.startswith('comment'):
                continue

            parts = line.split()
            if parts[0] == 'format':
                ply_format = parts[1]
            elif parts[0] == 'element':
                current_element = parts[1]
                if current_element == 'vertex':
                    num_verts = int(parts[2])
                elif current_element == 'face':
                    num_faces = int(parts[2])
            elif parts[0] == 'property':
                if current_element == 'vertex':
                    prop_type, prop_name = parts[1], parts[2]
                    vert_props.append((prop_name, prop_type))
                elif current_element == 'face':
                    if parts[1] == 'list':
                        face_list_type = parts[2]
                        face_index_type = parts[3]
            elif parts[0] == 'end_header':
                break

        # 2. Read vertices
        vertices = []
        if 'ascii' in ply_format:
            for i in range(num_verts):
                v_line = f.readline().decode('ascii', errors='ignore').strip().split()
                v_dict = {'id': i}
                for (pname, _), val_str in zip(vert_props, v_line):
                    v_dict[pname] = float(val_str)
                vertices.append(v_dict)
        else:
            fmt_str = '<' + ''.join(TYPE_MAP[ptype][0] for _, ptype in vert_props)
            row_size = struct.calcsize(fmt_str)
            for i in range(num_verts):
                v_bytes = f.read(row_size)
                vals = struct.unpack(fmt_str, v_bytes)
                v_dict = {'id': i}
                for (pname, _), val in zip(vert_props, vals):
                    v_dict[pname] = val
                vertices.append(v_dict)

        # 3. Read faces
        faces = []
        if 'ascii' in ply_format:
            for i in range(num_faces):
                f_line = f.readline().decode('ascii', errors='ignore').strip().split()
                if not f_line:
                    continue
                n_v = int(f_line[0])
                indices = [int(x) for x in f_line[1:1+n_v]]
                faces.append({'id': i, 'indices': indices})
        else:
            cnt_fmt, _ = TYPE_MAP[face_list_type]
            idx_fmt, idx_size = TYPE_MAP[face_index_type]
            for i in range(num_faces):
                cnt_bytes = f.read(struct.calcsize(cnt_fmt))
                if not cnt_bytes:
                    break
                n_v = struct.unpack('<' + cnt_fmt, cnt_bytes)[0]
                idx_bytes = f.read(n_v * idx_size)
                indices = list(struct.unpack(f'<{n_v}{idx_fmt}', idx_bytes))
                faces.append({'id': i, 'indices': indices})

    # 4. Compute statistics
    xs = [v['x'] for v in vertices]
    ys = [v['y'] for v in vertices]
    zs = [v['z'] for v in vertices]

    centroid = [sum(xs)/num_verts, sum(ys)/num_verts, sum(zs)/num_verts] if num_verts > 0 else [0.0, 0.0, 0.0]
    bbox = {
        'x': (min(xs), max(xs)) if xs else (0.0, 0.0),
        'y': (min(ys), max(ys)) if ys else (0.0, 0.0),
        'z': (min(zs), max(zs)) if zs else (0.0, 0.0),
    }

    info = {
        'path': str(ply_path),
        'format': ply_format,
        'num_vertices': num_verts,
        'num_faces': num_faces,
        'properties': [p[0] for p in vert_props],
        'centroid': centroid,
        'bbox': bbox
    }

    return vertices, faces, info


def expand_ply_to_text(ply_path, out_file=None):
    """
    Expand all vertices and faces of a PLY file into clean, formatted text.
    """
    vertices, faces, info = parse_ply(ply_path)

    lines = []
    lines.append("=" * 70)
    lines.append(f"PLY File Path: {info['path']}")
    lines.append(f"Format: {info['format']}")
    lines.append(f"Total Vertices: {info['num_vertices']}")
    lines.append(f"Total Faces:    {info['num_faces']}")
    c = info['centroid']
    lines.append(f"Geometric Centroid (Part Center p_k): x={c[0]:.4f}, y={c[1]:.4f}, z={c[2]:.4f}")
    bx, by, bz = info['bbox']['x'], info['bbox']['y'], info['bbox']['z']
    lines.append("Bounding Box (AABB):")
    lines.append(f"  X: [{bx[0]:.4f}, {bx[1]:.4f}] (Extent: {bx[1]-bx[0]:.4f})")
    lines.append(f"  Y: [{by[0]:.4f}, {by[1]:.4f}] (Extent: {by[1]-by[0]:.4f})")
    lines.append(f"  Z: [{bz[0]:.4f}, {bz[1]:.4f}] (Extent: {bz[1]-bz[0]:.4f})")
    lines.append("=" * 70)
    lines.append("")

    # All vertices
    lines.append(f"[ All Vertices ({len(vertices)} points) ]")
    has_normals = 'nx' in info['properties'] and 'ny' in info['properties'] and 'nz' in info['properties']
    for v in vertices:
        v_str = f"Vertex {v['id']:4d}: x={v['x']:8.4f}, y={v['y']:8.4f}, z={v['z']:8.4f}"
        if has_normals:
            v_str += f" | Normal: (nx={v['nx']:7.4f}, ny={v['ny']:7.4f}, nz={v['nz']:7.4f})"
        lines.append(v_str)

    lines.append("")
    # All faces
    lines.append(f"[ All Faces ({len(faces)} faces) ]")
    for f in faces:
        idx_str = ", ".join(f"{idx:4d}" for idx in f['indices'])
        lines.append(f"Face {f['id']:4d}: vertices [{idx_str}]")

    text_content = "\n".join(lines) + "\n"

    if out_file:
        out_p = Path(out_file).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, 'w', encoding='utf-8') as out:
            out.write(text_content)
        print(f"Successfully exported all vertices and faces to: {out_p}")
    else:
        print(text_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Expand and dump all vertices and faces of any 3D PLY file into readable text."
    )
    parser.add_argument(
        'ply_path',
        nargs='?',
        default=None,
        type=str,
        help="Path to the PLY file (positional: e.g. python utils/expand_ply.py path/to/file.ply)"
    )
    parser.add_argument(
        '--ply_path', '--ply',
        dest='ply_path_flag',
        type=str,
        default=None,
        help="Path to the PLY file (optional flag: e.g. --ply_path path/to/file.ply)"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help="Path to save the output text file (optional; defaults to printing to stdout)"
    )

    args = parser.parse_args()
    target_ply = args.ply_path_flag or args.ply_path
    if not target_ply:
        parser.error("Please provide a PLY file path (e.g., python utils/expand_ply.py path/to/model.ply or --ply_path path/to/model.ply)")

    expand_ply_to_text(target_ply, args.output)
