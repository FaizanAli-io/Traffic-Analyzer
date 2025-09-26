"""CLI entry point for the analyzer package.

Usage:
  python -m backend.analyzer --direction-orientation 0

This intentionally mirrors `backend/detr_motion.py`'s main function behavior
without modifying that file, and uses the pipeline adapter to invoke it.
"""

import argparse
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="GPU-Optimized DETR Vehicle Detection (Analyzer facade)"
    )
    parser.add_argument(
        "--direction-orientation",
        type=int,
        default=0,
        help="Direction orientation (0=North up, 1=East up, 2=South up, 3=West up)",
    )
    # Optional overrides kept minimal to match original CLI simplicity
    parser.add_argument("--video", type=str, default="input_video.mp4")
    parser.add_argument("--output", type=str, default="processed.mp4")

    args = parser.parse_args()

    run(
        video_path=args.video,
        output_path=args.output,
        direction_orientation=args.direction_orientation,
    )


if __name__ == "__main__":
    main()
