"""Flat pipeline adapter for running the analyzer."""

from .detection import GPUOptimizedDETRDetector


def run(
    direction_orientation: int = 0,
    video_path: str = "input_video_4.mp4",
    output_path: str = "output_detr_motion_filtered.mp4",
    confidence_threshold: float = 0.75,
    batch_size: int = 4,
    enable_mixed_precision: bool = True,
    display: bool = False,
    target_fps: int = 10,
    save_preview_frames: bool = False,
    preview_interval: int = 25,
):
    detector = GPUOptimizedDETRDetector(
        model_name="facebook/detr-resnet-101-dc5",
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        enable_mixed_precision=enable_mixed_precision,
    )

    detector.tracker.direction_manager.current_orientation = direction_orientation

    detector.process_video(
        video_path=video_path,
        output_path=output_path,
        display=display,
        target_fps=target_fps,
        save_preview_frames=save_preview_frames,
        preview_interval=preview_interval,
    )
