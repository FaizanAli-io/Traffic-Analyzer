"""Motion detection utilities (moved from detr_motion.py)."""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple


class MotionDetector:
    """Detect movement to filter out stationary objects"""

    def __init__(self, movement_threshold: float = 70, min_frames_to_confirm: int = 5):
        self.movement_threshold = movement_threshold
        self.min_frames_to_confirm = min_frames_to_confirm
        self.candidate_objects: Dict[int, List[Tuple[int, int]]] = {}
        self.confirmed_moving_objects: Set[int] = set()

    def calculate_movement(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate total movement from position history"""
        if len(positions) < 2:
            return 0

        total_movement = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            movement = math.sqrt(dx * dx + dy * dy)
            total_movement += movement

        return total_movement

    def is_moving(
        self,
        object_id: int,
        current_position: Tuple[int, int],
        position_history: List[Tuple[int, int]],
    ) -> bool:
        """Determine if an object is actually moving"""
        if object_id in self.confirmed_moving_objects:
            return True

        if object_id not in self.candidate_objects:
            self.candidate_objects[object_id] = [current_position]
            return False

        self.candidate_objects[object_id].append(current_position)

        if len(self.candidate_objects[object_id]) > 10:
            self.candidate_objects[object_id] = self.candidate_objects[object_id][-10:]

        if len(self.candidate_objects[object_id]) >= self.min_frames_to_confirm:
            total_movement = self.calculate_movement(self.candidate_objects[object_id])

            if total_movement > self.movement_threshold:
                self.confirmed_moving_objects.add(object_id)
                print(
                    f"  ✓ Confirmed movement for Object ID {object_id} (total movement: {total_movement:.1f}px)"
                )
                return True

        return False

    def cleanup_stale_candidates(self, active_object_ids: List[int]) -> None:
        """Remove candidates that are no longer being tracked"""
        stale_ids = set(self.candidate_objects.keys()) - set(active_object_ids)
        for stale_id in stale_ids:
            del self.candidate_objects[stale_id]
            self.confirmed_moving_objects.discard(stale_id)


__all__ = ["MotionDetector"]
