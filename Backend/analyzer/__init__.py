"""
Analyzer package (flat structure)

This package now hosts the refactored implementations originally found in
`backend/detr_motion.py`. We re-export the key classes directly from the flat
modules to avoid any circular dependencies and to enable eventually removing
`detr_motion.py` if desired.

Run the analyzer via:
  python -m backend.analyzer --direction-orientation 0

Modules:
- analyzer.detection -> GPUOptimizedDETRDetector
- analyzer.motion -> MotionDetector
- analyzer.ocr -> TimestampExtractor
- analyzer.tracking -> EnhancedObjectTracker, DirectionManager
- analyzer.pipeline -> run() convenience wrapper
"""

# Re-export key classes at top-level for convenience (from flat modules)
from .ocr import TimestampExtractor
from .motion import MotionDetector
from .tracking import EnhancedObjectTracker, DirectionManager
from .detection import GPUOptimizedDETRDetector

__all__ = [
    "TimestampExtractor",
    "MotionDetector",
    "EnhancedObjectTracker",
    "DirectionManager",
    "GPUOptimizedDETRDetector",
]
