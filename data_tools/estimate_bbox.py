import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_bbox(img, padding=0.15):
    """
    Estimate object bounding box for a mostly white-background image.

    Args:
        img:
            OpenCV BGR image, shape [H, W, 3]

        padding:
            Extra padding ratio around detected bbox.

    Returns:
        bbox:
            [x1, y1, x2, y2]

        foreground:
            Binary foreground mask, uint8, values {0, 255}

        candidates:
            List of connected-component information.
    """

    H, W = img.shape[:2]
    image_area = H * W

    # ======================================================
    # 1. Convert image to grayscale
    # ======================================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # ======================================================
    # 2. Automatically determine threshold using Otsu
    #
    # White background:
    #   background -> 0
    #   darker foreground -> 255
    # ======================================================

    otsu_threshold, foreground = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    print(f"Otsu threshold: {otsu_threshold}")

    # ======================================================
    # 3. Remove a tiny image border
    #
    # This prevents JPEG / resize / interpolation artifacts
    # around the outer edge from connecting to foreground.
    # ======================================================

    border = 2

    foreground[:border, :] = 0
    foreground[-border:, :] = 0
    foreground[:, :border] = 0
    foreground[:, -border:] = 0

    # ======================================================
    # 4. Morphological cleanup
    # ======================================================

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    # Remove isolated pixels
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Fill tiny holes / connect nearby pixels
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # ======================================================
    # 5. Connected-component analysis
    # ======================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )
    )

    candidates = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        bbox_area = w * h

        area_ratio = (
            area / image_area
        )

        bbox_ratio = (
            bbox_area / image_area
        )

        fill_ratio = (
            area / bbox_area
            if bbox_area > 0
            else 0.0
        )

        # ----------------------------------------------
        # Basic noise rejection
        # ----------------------------------------------

        if area < 10:
            continue

        # If the bounding box covers almost the whole image,
        # it is almost certainly a failed foreground region.
        if bbox_ratio > 0.80:
            continue

        candidates.append(
            {
                "label": int(i),

                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),

                "area": int(area),

                "area_ratio": float(area_ratio),
                "bbox_ratio": float(bbox_ratio),
                "fill_ratio": float(fill_ratio),
            }
        )

    if len(candidates) == 0:
        raise RuntimeError(
            "No valid foreground component found."
        )

    # ======================================================
    # 6. Select largest valid connected component
    # ======================================================

    best = max(
        candidates,
        key=lambda c: c["area"],
    )

    x = best["x"]
    y = best["y"]
    w = best["w"]
    h = best["h"]

    print()
    print("==========================")
    print("Selected component")
    print("==========================")

    print(
        f"x={x}, y={y}, "
        f"w={w}, h={h}, "
        f"area={best['area']}, "
        f"area_ratio={best['area_ratio']:.3f}, "
        f"bbox_ratio={best['bbox_ratio']:.3f}, "
        f"fill_ratio={best['fill_ratio']:.3f}"
    )

    # ======================================================
    # 7. Add padding
    # ======================================================

    pad_x = int(
        round(w * padding)
    )

    pad_y = int(
        round(h * padding)
    )

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        W - 1,
        x + w - 1 + pad_x
    )

    y2 = min(
        H - 1,
        y + h - 1 + pad_y
    )

    bbox = [
        int(x1),
        int(y1),
        int(x2),
        int(y2),
    ]

    return (
        bbox,
        foreground,
        candidates,
        float(otsu_threshold),
    )


def main():

    # ======================================================
    # Arguments
    # ======================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path",
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Extra padding ratio around detected object",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for bbox preview",
    )

    args = parser.parse_args()

    # ======================================================
    # Load image
    # ======================================================

    image_path = Path(
        args.image
    ).resolve()

    img = cv2.imread(
        str(image_path)
    )

    if img is None:
        raise RuntimeError(
            f"Cannot read image: {image_path}"
        )

    H, W = img.shape[:2]

    print()
    print("==========================")
    print("Input image")
    print("==========================")

    print(f"Path: {image_path}")
    print(f"Image size: W={W}, H={H}")

    # ======================================================
    # Estimate bbox
    # ======================================================

    (
        bbox,
        foreground,
        candidates,
        otsu_threshold,
    ) = estimate_bbox(
        img,
        padding=args.padding,
    )

    x1, y1, x2, y2 = bbox

    # ======================================================
    # Print candidate components
    # ======================================================

    print()
    print("==========================")
    print("Candidate components")
    print("==========================")

    sorted_candidates = sorted(
        candidates,
        key=lambda c: c["area"],
        reverse=True,
    )

    for c in sorted_candidates[:10]:

        print(
            f"label={c['label']:3d}  "
            f"area={c['area']:5d}  "
            f"area_ratio={c['area_ratio']:.3f}  "
            f"bbox_ratio={c['bbox_ratio']:.3f}  "
            f"fill={c['fill_ratio']:.3f}  "
            f"bbox=("
            f"{c['x']}, "
            f"{c['y']}, "
            f"{c['w']}, "
            f"{c['h']}"
            f")"
        )

    # ======================================================
    # Print result
    # ======================================================

    box_width = (
        x2 - x1 + 1
    )

    box_height = (
        y2 - y1 + 1
    )

    box_area = (
        box_width * box_height
    )

    box_area_ratio = (
        box_area / (W * H)
    )

    print()
    print("==========================")
    print("Estimated bounding box")
    print("==========================")

    print(
        "x1 y1 x2 y2:"
    )

    print(
        x1,
        y1,
        x2,
        y2
    )

    print(
        f"BBox size: "
        f"{box_width} x {box_height}"
    )

    print(
        f"BBox area ratio: "
        f"{box_area_ratio:.3f}"
    )

    print()
    print("SAM2 argument:")

    print(
        f"--box "
        f"{x1} "
        f"{y1} "
        f"{x2} "
        f"{y2}"
    )

    # ======================================================
    # Determine output paths
    # ======================================================

    if args.output is None:

        output_path = (
            image_path.parent.parent
            / "bbox_preview.png"
        )

    else:

        output_path = Path(
            args.output
        ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # Save bbox preview
    # ======================================================

    preview = img.copy()

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2,
    )

    # Add bbox text
    text = (
        f"[{x1}, {y1}, {x2}, {y2}]"
    )

    text_y = max(
        12,
        y1 - 5
    )

    cv2.putText(
        preview,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(output_path),
        preview,
    )

    # ======================================================
    # Save foreground debug image
    # ======================================================

    # foreground_path = (
    #     output_path.parent
    #     / "bbox_foreground_debug.png"
    # )

    # # foreground is already {0, 255}.
    # # DO NOT multiply by 255 again.
    # cv2.imwrite(
    #     str(foreground_path),
    #     foreground,
    # )

    # ======================================================
    # Save JSON metadata
    # ======================================================

    json_path = (
        output_path.with_suffix(
            ".json"
        )
    )

    metadata = {
        "image": str(image_path),

        "image_width": int(W),
        "image_height": int(H),

        "otsu_threshold": float(
            otsu_threshold
        ),

        "padding": float(
            args.padding
        ),

        "bbox_xyxy": [
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        ],

        "bbox_width": int(
            box_width
        ),

        "bbox_height": int(
            box_height
        ),

        "bbox_area_ratio": float(
            box_area_ratio
        ),

        "selected_component": {
            "x": int(
                sorted_candidates[0]["x"]
            ),
            "y": int(
                sorted_candidates[0]["y"]
            ),
            "w": int(
                sorted_candidates[0]["w"]
            ),
            "h": int(
                sorted_candidates[0]["h"]
            ),
            "area": int(
                sorted_candidates[0]["area"]
            ),
            "area_ratio": float(
                sorted_candidates[0]["area_ratio"]
            ),
            "bbox_ratio": float(
                sorted_candidates[0]["bbox_ratio"]
            ),
            "fill_ratio": float(
                sorted_candidates[0]["fill_ratio"]
            ),
        },
    }

    # with open(
    #     json_path,
    #     "w"
    # ) as f:

    #     json.dump(
    #         metadata,
    #         f,
    #         indent=4,
    #     )

    # ======================================================
    # Final output
    # ======================================================

    # print()
    # print("==========================")
    # print("Saved")
    # print("==========================")

    # print(
    #     f"BBox preview:     "
    #     f"{output_path}"
    # )

    # print(
    #     f"Foreground debug: "
    #     f"{foreground_path}"
    # )

    # print(
    #     f"BBox JSON:        "
    #     f"{json_path}"
    # )


if __name__ == "__main__":
    main()