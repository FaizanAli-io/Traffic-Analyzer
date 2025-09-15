# Analyzer facade (non-invasive)

This package provides a modular facade over the existing `backend/detr_motion.py` without changing its logic. It simply re-exports the original classes and offers a CLI wrapper so you can run the analyzer as a package.

Why: You asked to break `detr_motion.py` into multiple files but not change the original file or its logic. This structure enables incremental refactors later while keeping behavior identical today.

## Run

- From the repo root or `backend/` directory:

```
python -m backend.analyzer --direction-orientation 0
```

Options:

- `--direction-orientation`: 0=North up, 1=East up, 2=South up, 3=West up (defaults to 0)
- `--video`: input video path (defaults to `input_video_4.mp4`)
- `--output`: output video path (defaults to `output_detr_motion_filtered.mp4`)

## Modules (flat)

- `analyzer.detection` → `GPUOptimizedDETRDetector`
- `analyzer.motion` → `MotionDetector`
- `analyzer.ocr` → `TimestampExtractor`
- `analyzer.tracking` → `EnhancedObjectTracker`, `DirectionManager`
- `analyzer.pipeline` → Convenience `run()` adapter

Internally these modules import from `backend.detr_motion` so they stay in perfect sync.

## Notes

- No changes were made to `backend/detr_motion.py`.
- Future refactors can progressively move logic into these modules; today they simply delegate.
