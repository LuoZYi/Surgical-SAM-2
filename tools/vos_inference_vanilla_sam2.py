import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SAM2_REPO = Path(
    os.environ.get("ORIGINAL_SAM2_REPO", "/mnt/e/sam_2/sam2")
).resolve()
HELPER_SCRIPT = REPO_ROOT / "tools" / "vos_inference.py"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


if not ORIGINAL_SAM2_REPO.is_dir():
    raise FileNotFoundError(
        f"Original SAM2 repo not found: {ORIGINAL_SAM2_REPO}"
    )
if not HELPER_SCRIPT.is_file():
    raise FileNotFoundError(f"Helper script not found: {HELPER_SCRIPT}")

_prepend_sys_path(REPO_ROOT)
_prepend_sys_path(ORIGINAL_SAM2_REPO)

from sam2.build_sam import build_sam2_video_predictor
from sav_dataset.utils.endo_sav_benchmark import benchmark


def _load_helper_module():
    spec = importlib.util.spec_from_file_location(
        "surgical_vos_inference_helpers", HELPER_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper module from {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helper_module()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="Original SAM 2 model configuration name",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_small.pt",
        help="Path to the original SAM 2 checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="Directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="Directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="Text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="Directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="Threshold for the output mask logits",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="Use all available PNG files in input_mask_dir instead of only the first one per object",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="Apply original SAM 2 postprocessing such as hole filling",
    )
    parser.add_argument(
        "--first_frame_prompt_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gt_root",
        required=True,
        help="Path to the GT folder",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--read_frame_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_frame_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-n",
        "--num_processes",
        default=16,
        type=int,
        help="Number of concurrent evaluation processes",
    )
    parser.add_argument(
        "-s",
        "--strict",
        action="store_true",
        help="Require every GT video to have a corresponding prediction video",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress evaluation output",
    )
    parser.add_argument(
        "--do_not_skip_first_and_last_frame",
        action="store_true",
        help="Evaluate first and last annotated frames as well",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    print(f"Using GPU {args.gpu_id}")
    print(
        "Warning: only support evaluating one object per sequence and saving in object directories."
    )
    print("Testing with first frame prompt file:", args.first_frame_prompt_file)
    print("Original SAM2 repo:", ORIGINAL_SAM2_REPO)

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
    )

    print("Predictor class:", type(predictor))
    print("Predictor file:", inspect.getfile(type(predictor)))
    print("Predictor base class:", type(predictor).__mro__[1])

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )

    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    first_frame_prompts = helpers.load_first_frame_prompt_file(
        args.first_frame_prompt_file
    )

    total_pure_inference_time_sec = 0.0
    total_pure_inference_frames = 0
    max_peak_allocated_gb = 0.0
    max_peak_reserved_gb = 0.0

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        video_stats = helpers.vos_separate_inference_per_object(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            first_frame_prompts=(
                first_frame_prompts.get(video_name)
                if first_frame_prompts is not None
                else None
            ),
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            read_frame_interval=args.read_frame_interval,
            save_frame_interval=args.save_frame_interval,
        )
        total_pure_inference_time_sec += video_stats["pure_inference_time_sec"]
        total_pure_inference_frames += video_stats["pure_inference_frames"]
        max_peak_allocated_gb = max(
            max_peak_allocated_gb, video_stats["peak_cuda_memory_allocated_gb"]
        )
        max_peak_reserved_gb = max(
            max_peak_reserved_gb, video_stats["peak_cuda_memory_reserved_gb"]
        )

    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )
    total_pure_inference_fps = (
        total_pure_inference_frames / total_pure_inference_time_sec
        if total_pure_inference_time_sec > 0
        else 0.0
    )
    print(
        f"[PureInference][Summary] videos={len(video_names)} "
        f"frames={total_pure_inference_frames} "
        f"time={total_pure_inference_time_sec:.2f}s "
        f"fps={total_pure_inference_fps:.2f} "
        f"max_peak_alloc={max_peak_allocated_gb:.2f}GB "
        f"max_peak_reserved={max_peak_reserved_gb:.2f}GB"
    )

    epoch = args.sam2_checkpoint.split("/")[-1].split(".")[0].split("_")[-1]
    epoch = int(epoch) if epoch.isdigit() else 0

    benchmark(
        [args.gt_root],
        [args.output_mask_dir],
        args.strict,
        args.num_processes,
        verbose=not args.quiet,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        epoch=epoch,
    )


if __name__ == "__main__":
    main()
