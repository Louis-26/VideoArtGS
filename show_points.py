#!/usr/bin/env python3
import sys
import struct
import argparse

def dump_points(ply_path, out_path=None, limit=None):
    with open(ply_path, 'rb') as f:
        nv = 0
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                nv = int(line.split()[-1])
            elif line == 'end_header':
                break
        
        lines = []
        n_to_read = min(nv, limit) if limit else nv
        for i in range(n_to_read):
            v_data = f.read(24)
            x, y, z, nx, ny, nz = struct.unpack('<6f', v_data)
            lines.append(f"点 {i:3d}: x={x:8.4f}, y={y:8.4f}, z={z:8.4f}")

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as out:
            out.write(f"# {ply_path} 共 {nv} 个顶点\n")
            out.write("\n".join(lines) + "\n")
        print(f"已成功导出 {n_to_read} 个顶点到: {out_path}")
    else:
        print(f"# {ply_path} (共 {nv} 个顶点):")
        for l in lines[:50]:
            print(l)
        if nv > 50 and not limit:
            print(f"... (还有 {nv - 50} 个点，可使用 -o 保存为文件查看全部)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='一键解析 PLY 顶点坐标为文本列表')
    parser.add_argument('ply_file', help='PLY 文件路径')
    parser.add_argument('-o', '--output', help='输出 txt 文件路径 (可选)')
    parser.add_argument('-n', '--limit', type=int, help='限制查看前 N 个点 (可选)')
    args = parser.parse_args()
    dump_points(args.ply_file, args.output, args.limit)
