#!/usr/bin/env python3
"""
Backfill joint_state_error into existing videoartgs result.csv files.

eval.py now computes the state error during evaluation (from gt/part_info.json);
this script adds it to result.csv files produced before that change, without
re-running the full eval. The CD-based joint matching (perm) is recomputed from
the saved part meshes replaying eval.py's exact call order under the same seed,
so the pairing is identical to the one the original eval used.

Needs a GPU (compute_chamfer runs on CUDA).

Usage:
    python utils/backfill_state_error.py --subset sapien
    python utils/backfill_state_error.py --subset sapien --scenes 168 45612 --force
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import (read_gt, find_eval_perm_cd, compute_recon_error,
                           read_part_info_states, joint_state_metric, seed_everything)


def replay_perm(gt_path, mesh_path, num_d_joints):
    """Reproduce the perm computed inside eval_CD's first trial: same seed and
    same preceding compute_recon_error calls, so the RNG stream matches."""
    if num_d_joints == 1:
        return [0]
    seed_everything(0)
    for pred_ply, gt_ply in [(f'{mesh_path}/part_0.ply', f'{gt_path}/part_0.ply'),
                             (f'{mesh_path}/whole_mesh.ply', f'{gt_path}/whole_mesh.ply')]:
        try:
            compute_recon_error(pred_ply, gt_ply, n_samples=10000, vis=False)
        except Exception:
            pass
    _, perm = find_eval_perm_cd(gt_path, mesh_path, num_d_joints)
    return perm


def backfill_scene(source_path, save_dir, force=False):
    part_info_path = os.path.join(source_path, 'gt', 'part_info.json')
    joint_value_path = os.path.join(save_dir, 'joint_value.npy')
    result_path = os.path.join(save_dir, 'result.csv')
    for p, what in [(part_info_path, 'part_info.json'), (joint_value_path, 'joint_value.npy'),
                    (result_path, 'result.csv')]:
        if not os.path.exists(p):
            return f'skip (no {what})'

    df = pd.read_csv(result_path, index_col='Metric')
    if 'joint_state_error' in df.index and not force:
        return 'skip (already has joint_state_error)'

    gt_path = os.path.join(source_path, 'gt')
    gt_joint_list = read_gt(os.path.join(gt_path, 'mobility_v2.json'))
    perm = replay_perm(gt_path, os.path.join(save_dir, 'meshes'), len(gt_joint_list))

    theta = np.load(joint_value_path)
    n_dyn = theta.shape[1]
    gt_states = read_part_info_states(part_info_path, gt_joint_list)

    errs, types = [], []
    for i, gt_joint in enumerate(gt_joint_list):
        if i < len(perm):
            err = joint_state_metric(theta[perm[i] + 1],
                                     gt_states[gt_joint['idx']][-n_dyn:],
                                     gt_joint['joint_type'])
        else:
            err = 90. if gt_joint['joint_type'] == 'r' else 100.
        errs.append(err)
        types.append(gt_joint['joint_type'])
        df.loc[f'joint_state_error_{i}'] = err

    df.loc['joint_state_error'] = float(np.mean(errs))
    r_errs = [e for e, t in zip(errs, types) if t == 'r']
    p_errs = [e for e, t in zip(errs, types) if t == 'p']
    if r_errs:
        df.loc['joint_state_error_r'] = float(np.mean(r_errs))  # deg
    if p_errs:
        df.loc['joint_state_error_p'] = float(np.mean(p_errs))  # cm
    df.to_csv(result_path)
    detail = ', '.join(f'{t}:{e:.3f}' for e, t in zip(errs, types))
    return f'joint_state_error={np.mean(errs):.3f} (perm={perm}) [{detail}]'


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--base', default='.', help='Repo root (where outputs/ and data/ live).')
    ap.add_argument('--dataset', default='videoartgs')
    ap.add_argument('--subset', default='sapien')
    ap.add_argument('--output_dir', default='outputs')
    ap.add_argument('--iteration', default=20000, type=int)
    ap.add_argument('--scenes', nargs='*', default=None,
                    help='Optional explicit scene list; default = all with a result.csv.')
    ap.add_argument('--force', action='store_true',
                    help='Recompute even if joint_state_error is already present.')
    args = ap.parse_args()

    pattern = os.path.join(args.base, args.output_dir, args.dataset, args.subset,
                           '*', 'final', 'train', f'ours_{args.iteration}', 'result.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No result.csv found matching {pattern}')

    for p in paths:
        scene = p.split(os.sep)[-5]
        if args.scenes is not None and scene not in args.scenes:
            continue
        source_path = os.path.join(args.base, 'data', args.dataset, args.subset, scene)
        status = backfill_scene(source_path, os.path.dirname(p), force=args.force)
        print(f'{scene}: {status}')


if __name__ == '__main__':
    main()
