"""
PartField feature extraction wrapper, adapted from the official Particulate repo
(https://github.com/RuiningLi/particulate, partfield_utils.py).

Local changes vs upstream:
- PartField code lives at <repo_root>/PartField (sibling of this package).
- The checkpoint is loaded from particulate/model_ckpt/model_objaverse.ckpt
  instead of being downloaded from HuggingFace.
"""
import argparse
import os
import sys

import torch

_PARTFIELD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PartField"))
sys.path.append(_PARTFIELD_DIR)
from partfield.model.PVCNN.encoder_pc import sample_triplane_feat
from partfield.model_trainer_pvcnn_only_demo import Model
from partfield.config import setup


@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
def obtain_partfield_feats(
    partfield_model,
    points_enc,
    points_dec,
):
    bbmin = points_enc.min(dim=-2, keepdim=True)[0]
    bbmax = points_enc.max(dim=-2, keepdim=True)[0]
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.9 / (bbmax - bbmin).max()
    points_enc = (points_enc - center) * scale
    points_dec = (points_dec - center) * scale

    pc_feat = partfield_model.pvcnn(points_enc, points_enc)
    planes = partfield_model.triplane_transformer(pc_feat)
    sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
    point_feat = sample_triplane_feat(part_planes, points_dec)
    return point_feat


def get_partfield_model(device='cuda'):
    ckpt_path = os.path.join(os.path.dirname(__file__), "model_ckpt", "model_objaverse.ckpt")
    config_path = os.path.join(_PARTFIELD_DIR, "configs", "final", "demo.yaml")
    partfield_model = Model.load_from_checkpoint(
        ckpt_path,
        cfg=setup(argparse.Namespace(config_file=config_path, opts=[]), freeze=False)
    )
    partfield_model.eval()
    partfield_model.to(device=device)
    return partfield_model
