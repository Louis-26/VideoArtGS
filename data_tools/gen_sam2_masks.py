import argparse
import tempfile
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_video_predictor

# ---------------------------------------------------------
# Reuse the bbox estimator that we already tested
# data_tools/estimate_bbox.py
# ---------------------------------------------------------
try:
    from estimate_bbox import estimate_bbox
except ImportError:
    from data_tools.estimate_bbox import estimate_bbox


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="VideoArtGS images/ directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output masks/ directory"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )

    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame used to initialize SAM2"
    )

    # -----------------------------------------------------
    # Optional manual override
    #
    # If not supplied:
    # estimate_bbox.py will generate it automatically.
    # -----------------------------------------------------
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Optional manual bounding box [x1 y1 x2 y2]. "
            "If omitted, bbox is estimated automatically."
        )
    )

    # We already confirmed 0 padding works well for microwave
    parser.add_argument(
        "--bbox_padding",
        type=float,
        default=0.0,
        help="Padding used by automatic bbox estimation"
    )

    parser.add_argument(
        "--save_vis",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # 1. Read input frames
    # =====================================================

    image_paths = sorted(
        list(image_dir.glob("*.png")),
        key=lambda p: int(p.stem)
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No PNG images found in {image_dir}"
        )

    if (
        args.frame_idx < 0
        or args.frame_idx >= len(image_paths)
    ):
        raise ValueError(
            f"frame_idx={args.frame_idx} is out of range. "
            f"Total frames={len(image_paths)}"
        )

    print(f"Found {len(image_paths)} frames")

    prompt_image_path = image_paths[
        args.frame_idx
    ]

    # =====================================================
    # 2. Determine SAM2 bounding box
    #
    # manual --box
    #       OR
    # estimate_bbox.py
    # =====================================================

    if args.box is not None:

        box = np.array(
            args.box,
            dtype=np.float32
        )

        bbox_source = "manual"
        otsu_threshold = None

        print("")
        print("==============================")
        print("Using manual bounding box")
        print("==============================")
        print(f"box = {box.tolist()}")

    else:

        print("")
        print("==============================")
        print("Estimating bounding box")
        print("==============================")

        print(
            f"Frame: {args.frame_idx} "
            f"({prompt_image_path.name})"
        )

        # estimate_bbox expects OpenCV BGR image
        import cv2

        prompt_cv = cv2.imread(
            str(prompt_image_path)
        )

        if prompt_cv is None:
            raise RuntimeError(
                f"Cannot read prompt image: "
                f"{prompt_image_path}"
            )

        (
            bbox,
            foreground,
            candidates,
            otsu_threshold,
        ) = estimate_bbox(
            prompt_cv,
            padding=args.bbox_padding,
        )

        box = np.array(
            bbox,
            dtype=np.float32
        )

        bbox_source = "auto_otsu"

        print("")
        print("==============================")
        print("Automatic bbox result")
        print("==============================")

        print(
            f"Otsu threshold: "
            f"{otsu_threshold}"
        )

        print(
            f"box = {box.tolist()}"
        )

    # =====================================================
    # 3. Validate box
    # =====================================================

    with Image.open(prompt_image_path) as img:
        W, H = img.size

    x1, y1, x2, y2 = box.tolist()

    if not (
        0 <= x1 < x2 < W
        and
        0 <= y1 < y2 < H
    ):
        raise RuntimeError(
            f"Invalid bbox {box.tolist()} "
            f"for image size W={W}, H={H}"
        )

    bbox_width = x2 - x1 + 1
    bbox_height = y2 - y1 + 1

    bbox_area_ratio = (
        bbox_width
        * bbox_height
        / (W * H)
    )

    print(
        f"BBox area ratio: "
        f"{bbox_area_ratio:.3f}"
    )

    # =====================================================
    # 4. Save prompt preview BEFORE SAM2 inference
    #
    # This is useful because if the box is wrong we can
    # immediately inspect it.
    # =====================================================

    prompt_img = Image.open(
        prompt_image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(
        prompt_img
    )

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=2
    )

    # preview_path = (
    #     output_dir.parent
    #     / "sam2_prompt_preview.png"
    # )

    # prompt_img.save(
    #     preview_path
    # )

    # =====================================================
    # 5. Save bbox metadata
    # =====================================================

    # bbox_json = (
    #     output_dir.parent
    #     / "sam2_prompt_box.json"
    # )

    # metadata = {
    #     "frame_idx": int(args.frame_idx),
    #     "frame_name": prompt_image_path.name,

    #     "image_width": int(W),
    #     "image_height": int(H),

    #     "box_xyxy": [
    #         float(x1),
    #         float(y1),
    #         float(x2),
    #         float(y2),
    #     ],

    #     "box_source": bbox_source,

    #     "bbox_padding": (
    #         float(args.bbox_padding)
    #         if args.box is None
    #         else None
    #     ),

    #     "otsu_threshold": (
    #         float(otsu_threshold)
    #         if otsu_threshold is not None
    #         else None
    #     ),

    #     "bbox_area_ratio": float(
    #         bbox_area_ratio
    #     ),
    # }

    # with open(
    #     bbox_json,
    #     "w"
    # ) as f:
    #     json.dump(
    #         metadata,
    #         f,
    #         indent=4
    #     )

    # print(f"Prompt preview: {preview_path}")
    # print(f"BBox metadata:  {bbox_json}")

    # =====================================================
    # 6. Prepare JPEG video sequence for SAM2
    # =====================================================

    with tempfile.TemporaryDirectory(
        prefix="sam2_frames_"
    ) as tmp_dir:

        tmp_dir = Path(tmp_dir)

        print("")
        print("Preparing temporary JPEG sequence...")

        for i, image_path in enumerate(
            image_paths
        ):

            img = Image.open(
                image_path
            ).convert("RGB")

            jpg_path = (
                tmp_dir
                / f"{i:06d}.jpg"
            )

            img.save(
                jpg_path,
                quality=100,
                subsampling=0
            )

        # =================================================
        # 7. Load SAM2
        # =================================================

        print("Loading SAM2...")

        predictor = (
            build_sam2_video_predictor(
                args.config,
                str(checkpoint)
            )
        )

        print(
            "Initializing video state..."
        )

        inference_state = (
            predictor.init_state(
                video_path=str(tmp_dir)
            )
        )

        # =================================================
        # 8. Give SAM2 the initial object bbox
        # =================================================

        print(
            f"Adding object prompt "
            f"at frame {args.frame_idx}: "
            f"{box.tolist()}"
        )

        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16
            )
        ):

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=args.frame_idx,
                obj_id=0,
                box=box,
            )

            # =============================================
            # 9. Propagate object through entire video
            # =============================================

            object_masks = {}

            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits
            ) in predictor.propagate_in_video(
                inference_state
            ):

                for i, obj_id in enumerate(
                    out_obj_ids
                ):

                    obj_id = int(obj_id)

                    if obj_id != 0:
                        continue

                    mask = (
                        out_mask_logits[i]
                        > 0.0
                    )

                    mask = (
                        mask
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    # SAM2 normally returns [1,H,W]
                    if mask.ndim == 2:
                        mask = mask[None]

                    object_masks[
                        out_frame_idx
                    ] = mask

        # =================================================
        # 10. Save VideoArtGS-compatible masks
        #
        # Each frame:
        #
        # [2, 1, H, W]
        #
        # channel 0 = articulated object
        # channel 1 = human/hand (currently zero)
        # =================================================

        print("Saving masks...")

        if args.save_vis:

            vis_dir = (
                output_dir.parent
                / "masks_vis"
            )

            vis_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        for frame_idx, image_path in enumerate(
            image_paths
        ):

            if frame_idx not in object_masks:
                raise RuntimeError(
                    f"SAM2 did not return mask "
                    f"for frame {frame_idx}"
                )

            object_mask = (
                object_masks[frame_idx]
            )

            human_mask = np.zeros_like(
                object_mask,
                dtype=np.uint8
            )

            masks = np.stack(
                [
                    object_mask,
                    human_mask
                ],
                axis=0
            )

            # Shape:
            # [2, 1, H, W]
            np.save(
                output_dir
                / f"{frame_idx:06d}.npy",
                masks
            )

            # =============================================
            # Optional visualization
            # =============================================

            if args.save_vis:

                mask_2d = object_mask[0]

                vis = (
                    mask_2d * 255
                ).astype(np.uint8)

                Image.fromarray(
                    vis
                ).save(
                    vis_dir
                    / f"{frame_idx:06d}.png"
                )

    # =====================================================
    # Done
    # =====================================================

    print("")
    print("====================================")
    print("SAM2 segmentation finished")
    print("====================================")

    print(f"Masks:   {output_dir}")
    # print(f"Preview: {preview_path}")
    # print(f"BBox:    {bbox_json}")
    print(f"Frames:  {len(image_paths)}")

    print("")
    print("Each mask shape:")
    print("[2, 1, H, W]")

    print(
        "channel 0 = articulated object"
    )

    print(
        "channel 1 = human/hand "
        "(currently zero)"
    )


if __name__ == "__main__":
    main()