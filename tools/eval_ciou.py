import argparse
import os
from os import path

import numpy as np
from PIL import Image


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    return x1, y1, x2, y2


def bbox_iou_xyxy(box1, box2, inclusive: bool = True) -> float:
    offset = 1.0 if inclusive else 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1 + offset)
    inter_h = max(0.0, y2 - y1 + offset)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0] + offset) * max(0.0, box1[3] - box1[1] + offset)
    area2 = max(0.0, box2[2] - box2[0] + offset) * max(0.0, box2[3] - box2[1] + offset)
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_ciou_xyxy(box1, box2, inclusive: bool = True) -> float:
    offset = 1.0 if inclusive else 0.0
    iou = bbox_iou_xyxy(box1, box2, inclusive=inclusive)

    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    enc_x1 = min(box1[0], box2[0])
    enc_y1 = min(box1[1], box2[1])
    enc_x2 = max(box1[2], box2[2])
    enc_y2 = max(box1[3], box2[3])
    c_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    w1 = max(offset, box1[2] - box1[0] + offset)
    h1 = max(offset, box1[3] - box1[1] + offset)
    w2 = max(offset, box2[2] - box2[0] + offset)
    h2 = max(offset, box2[3] - box2[1] + offset)

    v = (4.0 / (np.pi ** 2)) * (np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2
    alpha = v / max(1e-12, 1.0 - iou + v)
    distance_term = 0.0 if c_diag_sq <= 0.0 else center_dist_sq / c_diag_sq
    ciou = iou - distance_term - alpha * v
    return float(max(-1.0, min(1.0, ciou)))


def frame_ciou(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    object_id: int,
    *,
    inclusive_bbox: bool = True,
    empty_policy: str = "one_zero",
):
    gt_obj = gt_mask == object_id
    pred_obj = pred_mask == object_id

    gt_box = mask_to_bbox(gt_obj)
    pred_box = mask_to_bbox(pred_obj)

    if gt_box is None and pred_box is None:
        if empty_policy == "one_zero":
            return 1.0
        if empty_policy == "zero_zero":
            return 0.0
        raise ValueError(f"Unsupported empty_policy={empty_policy!r}")
    if gt_box is None or pred_box is None:
        return 0.0
    return bbox_ciou_xyxy(gt_box, pred_box, inclusive=inclusive_bbox)


def scan_video_objects(gt_root: str, pred_root: str, video_name: str):
    pred_video_dir = path.join(pred_root, video_name)
    gt_video_dir = path.join(gt_root, video_name)
    entries = sorted(os.listdir(pred_video_dir))
    objects = []
    for entry in entries:
        obj_dir = path.join(pred_video_dir, entry)
        if not entry.isdigit() or not path.isdir(obj_dir):
            continue
        frames = sorted(
            [f for f in os.listdir(obj_dir) if f.lower().endswith(".png")]
        )
        frames = [f for f in frames if path.exists(path.join(gt_video_dir, f))]
        objects.append((int(entry), frames, gt_video_dir, obj_dir))
    return objects


def evaluate_video(
    gt_root: str,
    pred_root: str,
    video_name: str,
    skip_first_and_last: bool,
    *,
    inclusive_bbox: bool = True,
    empty_policy: str = "one_zero",
):
    object_scores = {}
    for object_id, frames, gt_dir, pred_dir in scan_video_objects(gt_root, pred_root, video_name):
        eval_frames = frames[1:-1] if skip_first_and_last else frames
        if not eval_frames:
            continue
        scores = []
        for frame in eval_frames:
            gt_mask = np.array(Image.open(path.join(gt_dir, frame)))
            pred_mask = np.array(Image.open(path.join(pred_dir, frame)))
            scores.append(
                frame_ciou(
                    gt_mask,
                    pred_mask,
                    object_id,
                    inclusive_bbox=inclusive_bbox,
                    empty_policy=empty_policy,
                )
            )
        object_scores[object_id] = float(np.mean(scores) * 100.0)
    return object_scores


def evaluate_dataset(
    gt_root: str,
    pred_root: str,
    strict: bool,
    skip_first_and_last: bool,
    *,
    inclusive_bbox: bool = True,
    empty_policy: str = "one_zero",
    aggregate_by: str = "object",
):
    gt_videos = sorted([v for v in os.listdir(gt_root) if path.isdir(path.join(gt_root, v))])
    pred_videos = sorted([v for v in os.listdir(pred_root) if path.isdir(path.join(pred_root, v))])

    if strict:
        missing = sorted(set(gt_videos) - set(pred_videos))
        extra = sorted(set(pred_videos) - set(gt_videos))
        if missing:
            raise FileNotFoundError(f"Missing prediction videos: {missing}")
        if extra:
            raise FileNotFoundError(f"Extra prediction videos: {extra}")
        videos = gt_videos
    else:
        videos = sorted(set(gt_videos) & set(pred_videos))

    print(f"In dataset {gt_root}, evaluating {len(videos)} videos: {videos}")

    per_video = {}
    all_scores = []
    for video_name in videos:
        video_scores = evaluate_video(
            gt_root,
            pred_root,
            video_name,
            skip_first_and_last,
            inclusive_bbox=inclusive_bbox,
            empty_policy=empty_policy,
        )
        per_video[video_name] = video_scores
        if aggregate_by == "object":
            all_scores.extend(video_scores.values())
        elif aggregate_by == "video":
            if video_scores:
                all_scores.append(float(np.mean(list(video_scores.values()))))
        else:
            raise ValueError(f"Unsupported aggregate_by={aggregate_by!r}")

    global_ciou = float(np.mean(all_scores)) if all_scores else float("nan")
    return global_ciou, per_video


def format_report(global_ciou: float, per_video: dict) -> str:
    ml = max(*[len(n) for n in per_video.keys()], len("Global score")) if per_video else len("Global score")
    out = f'{"sequence":<{ml}},{"obj":>3}, {"CIoU":>8}\n'
    out += f'{"Global score":<{ml}},{"":>3}, {global_ciou:>8.2f}\n'
    for video_name, object_scores in per_video.items():
        for object_id in sorted(object_scores):
            out += f"{video_name:<{ml}},{object_id:03}, {object_scores[object_id]:>8.2f}\n"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--do_not_skip_first_and_last_frame", action="store_true")
    parser.add_argument(
        "--bbox_mode",
        choices=["inclusive", "exclusive"],
        default="inclusive",
    )
    parser.add_argument(
        "--aggregate_by",
        choices=["object", "video"],
        default="object",
    )
    parser.add_argument(
        "--empty_policy",
        choices=["one_zero", "zero_zero"],
        default="one_zero",
    )
    args = parser.parse_args()

    global_ciou, per_video = evaluate_dataset(
        gt_root=args.gt_root,
        pred_root=args.pred_root,
        strict=args.strict,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        inclusive_bbox=(args.bbox_mode == "inclusive"),
        empty_policy=args.empty_policy,
        aggregate_by=args.aggregate_by,
    )
    report = format_report(global_ciou, per_video)
    print(report, end="")

    result_path = path.join(args.pred_root, "ciou_results.csv")
    print(f"Saving the CIoU results to {result_path}")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
