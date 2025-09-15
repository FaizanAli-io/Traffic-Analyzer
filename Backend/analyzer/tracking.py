"""Tracking and direction management (moved from detr_motion.py)."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import time

from .motion import MotionDetector


class DirectionManager:
    """Manages frame orientation and direction mapping"""

    def __init__(self):
        # Base directions in order: [top, right, bottom, left]
        self.base_directions = ["North", "East", "South", "West"]
        self.current_orientation = 0  # Index pointing to what direction is at the top

    def rotate_clockwise(self):
        """Rotate directions clockwise: North->East, East->South, etc."""
        self.current_orientation = (self.current_orientation + 1) % 4
        print(f"Rotated clockwise. New orientation: {self.get_current_mapping()}")

    def rotate_counterclockwise(self):
        """Rotate directions counter-clockwise: North->West, West->South, etc."""
        self.current_orientation = (self.current_orientation - 1) % 4
        print(
            f"Rotated counter-clockwise. New orientation: {self.get_current_mapping()}"
        )

    def get_current_directions(self):
        """Get current direction mapping: [top, right, bottom, left]"""
        return [
            self.base_directions[self.current_orientation],  # top
            self.base_directions[(self.current_orientation + 1) % 4],  # right
            self.base_directions[(self.current_orientation + 2) % 4],  # bottom
            self.base_directions[(self.current_orientation + 3) % 4],  # left
        ]

    def get_current_mapping(self):
        """Get readable mapping of edges to directions"""
        directions = self.get_current_directions()
        return {
            "top": directions[0],
            "right": directions[1],
            "bottom": directions[2],
            "left": directions[3],
        }

    def determine_direction_from_position(
        self,
        centroid: Tuple[int, int],
        frame_width: int,
        frame_height: int,
        border_threshold: int = 50,
    ) -> str:
        """
        Determine direction using X-pattern zones and border detection
        Args:
            centroid: (x, y) position
            frame_width: frame width
            frame_height: frame height
            border_threshold: distance from edge to consider as border entry/exit
        Returns:
            direction string (North/South/East/West)
        """
        x, y = centroid

        # Calculate distances to each edge
        dist_to_top = y
        dist_to_bottom = frame_height - y
        dist_to_left = x
        dist_to_right = frame_width - x

        # Check if near any border (for entry/exit detection)
        near_top = dist_to_top <= border_threshold
        near_bottom = dist_to_bottom <= border_threshold
        near_left = dist_to_left <= border_threshold
        near_right = dist_to_right <= border_threshold

        # If not near any border, determine zone using X-pattern
        if not (near_top or near_bottom or near_left or near_right):
            # Use X-pattern to determine zone
            # center_x, center_y = frame_width // 2, frame_height // 2  # not used below

            # Determine which side of each diagonal the point is on
            # Diagonal 1: top-left to bottom-right (y = x * height/width)
            diagonal1_y = (x * frame_height) / frame_width
            above_diagonal1 = y < diagonal1_y

            # Diagonal 2: top-right to bottom-left (y = height - x * height/width)
            diagonal2_y = frame_height - (x * frame_height) / frame_width
            above_diagonal2 = y < diagonal2_y

            # Determine zone based on diagonal positions
            if above_diagonal1 and above_diagonal2:
                zone_index = 0  # North (top)
            elif above_diagonal1 and not above_diagonal2:
                zone_index = 1  # East (right)
            elif not above_diagonal1 and not above_diagonal2:
                zone_index = 2  # South (bottom)
            else:  # not above_diagonal1 and above_diagonal2
                zone_index = 3  # West (left)

            # Map zone to current direction
            current_directions = self.get_current_directions()
            return current_directions[zone_index]

        # If near border, determine direction based on closest edge
        distances = {
            "top": dist_to_top,
            "right": dist_to_right,
            "bottom": dist_to_bottom,
            "left": dist_to_left,
        }

        closest_edge = min(distances.keys(), key=lambda k: distances[k])

        # Map edge to current direction
        current_directions = self.get_current_directions()
        edge_to_direction = {
            "top": current_directions[0],  # North
            "right": current_directions[1],  # East
            "bottom": current_directions[2],  # South
            "left": current_directions[3],  # West
        }

        return edge_to_direction[closest_edge]


class EnhancedObjectTracker:
    """Advanced object tracker with IoU-based duplicate prevention and motion filtering"""

    def __init__(
        self,
        max_disappeared: int = 3,
        max_distance: int = 150,
        iou_threshold: float = 0.6,
    ):
        self.next_object_id = 0
        self.objects: Dict[int, dict] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold

        self.object_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.object_classes: Dict[int, int] = {}
        self.object_timestamps: Dict[int, dict] = {}
        self.object_durations: Dict[int, dict] = {}

        # NEW: Direction tracking
        self.direction_manager = DirectionManager()
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.object_directions: Dict[int, dict] = (
            {}
        )  # Store origin/destination for each object

        # Add motion detector
        self.motion_detector = MotionDetector(
            movement_threshold=20, min_frames_to_confirm=3
        )

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Update frame dimensions"""
        self.frame_width = width
        self.frame_height = height

    def calculate_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def remove_duplicate_detections(
        self,
        detections: List[Tuple[Tuple[int, int], int, float, Tuple[int, int, int, int]]],
    ):
        """Remove duplicate detections using IoU threshold"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        detections_with_idx = [(i, det) for i, det in enumerate(detections)]
        detections_with_idx.sort(key=lambda x: x[1][2], reverse=True)

        filtered_detections = []
        used_indices = set()

        for i, (orig_idx, detection) in enumerate(detections_with_idx):
            if orig_idx in used_indices:
                continue

            centroid, class_id, confidence, bbox = detection
            is_duplicate = False

            for existing_detection in filtered_detections:
                existing_bbox = existing_detection[3]
                existing_class = existing_detection[1]

                if existing_class == class_id:
                    iou = self.calculate_iou(bbox, existing_bbox)
                    if iou > self.iou_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered_detections.append(detection)
                used_indices.add(orig_idx)

        if len(detections) != len(filtered_detections):
            print(
                f"  Removed {len(detections) - len(filtered_detections)} duplicate detections"
            )
        return filtered_detections

    def register(
        self,
        centroid: Tuple[int, int],
        class_id: int,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        timestamp=None,
    ) -> int:
        """Register a new object with timestamp and origin direction"""
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "class_id": class_id,
            "confidence": confidence,
            "bbox": bbox,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "confirmed_moving": False,
        }
        self.disappeared[self.next_object_id] = 0
        self.object_classes[self.next_object_id] = class_id
        self.object_history[self.next_object_id].append(centroid)

        if timestamp:
            self.object_timestamps[self.next_object_id] = {
                "first_seen_timestamp": timestamp,
                "last_seen_timestamp": timestamp,
            }

        # NEW: Determine origin direction
        origin_direction = self.direction_manager.determine_direction_from_position(
            centroid, self.frame_width, self.frame_height
        )

        self.object_directions[self.next_object_id] = {
            "origin": origin_direction,
            "destination": None,
            "first_centroid": centroid,
            "last_centroid": centroid,
        }

        if origin_direction:
            print(f"  → Object {self.next_object_id} entered from {origin_direction}")

        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id: int) -> None:
        """Remove an object and calculate final duration + destination direction + CREATE CSV RECORD"""
        # Determine final destination direction
        if object_id in self.object_directions:
            last_centroid = self.object_directions[object_id]["last_centroid"]
            destination_direction = (
                self.direction_manager.determine_direction_from_position(
                    last_centroid, self.frame_width, self.frame_height
                )
            )

            # Update final destination
            if (
                destination_direction
                and destination_direction != self.object_directions[object_id]["origin"]
            ):
                self.object_directions[object_id]["destination"] = destination_direction

            origin = self.object_directions[object_id]["origin"]
            destination = self.object_directions[object_id]["destination"]

            if destination_direction:
                print(
                    f"  ← Object {object_id} exited to {destination_direction} (from {origin})"
                )

        # Calculate duration and CREATE CSV RECORD for confirmed moving objects
        if object_id in self.object_timestamps and self.objects.get(object_id, {}).get(
            "confirmed_moving", False
        ):

            timestamps = self.object_timestamps[object_id]
            first_seen = timestamps["first_seen_timestamp"]
            last_seen = timestamps["last_seen_timestamp"]

            if first_seen and last_seen:
                duration = (last_seen - first_seen).total_seconds()
                self.object_durations[object_id] = {
                    "class": self.object_classes.get(object_id, "unknown"),
                    "duration": duration,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                }
                print(
                    f"★ Moving Object {object_id} ({self.object_classes.get(object_id, 'unknown')}) "
                    f"completed: {duration:.2f}s visible "
                    f"({first_seen.strftime('%H:%M:%S')} → {last_seen.strftime('%H:%M:%S')})"
                )

            del self.object_timestamps[object_id]

            # Clean up from motion detector
            self.motion_detector.confirmed_moving_objects.discard(object_id)
            if object_id in self.motion_detector.candidate_objects:
                del self.motion_detector.candidate_objects[object_id]

            # Clean up main tracking data
            if object_id in self.objects:
                del self.objects[object_id]
            if object_id in self.disappeared:
                del self.disappeared[object_id]
            if object_id in self.object_classes:
                del self.object_classes[object_id]
            if object_id in self.object_history:
                del self.object_history[object_id]

            # NEW: Finalize detection record before cleanup
            if hasattr(self, "detector") and hasattr(
                self.detector, "finalize_detection_record"
            ):
                self.detector.finalize_detection_record(object_id)

    def calculate_distance(
        self, point_a: Tuple[int, int], point_b: Tuple[int, int]
    ) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(
            (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
        )

    def predict_next_position(self, object_id: int) -> Tuple[int, int]:
        """Predict next position based on movement history"""
        if (
            object_id not in self.object_history
            or len(self.object_history[object_id]) < 2
        ):
            return self.objects[object_id]["centroid"]

        history = self.object_history[object_id]
        if len(history) >= 3:
            recent_points = history[-3:]
            velocity_x = (recent_points[-1][0] - recent_points[0][0]) / 2
            velocity_y = (recent_points[-1][1] - recent_points[0][1]) / 2
        else:
            velocity_x = history[-1][0] - history[-2][0]
            velocity_y = history[-1][1] - history[-2][1]

        predicted_x = history[-1][0] + velocity_x
        predicted_y = history[-1][1] + velocity_y

        return (int(predicted_x), int(predicted_y))

    def update(self, detections, timestamp=None):
        """Update tracker with motion-filtered detections"""
        detections = self.remove_duplicate_detections(detections)

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        if len(self.objects) == 0:
            tracked_objects = {}
            for detection in detections:
                centroid, class_id, confidence, bbox = detection
                object_id = self.register(
                    centroid, class_id, confidence, bbox, timestamp
                )
                tracked_objects[object_id] = self.objects[object_id]
            return tracked_objects

        # Enhanced matching with both distance and IoU
        object_ids = list(self.objects.keys())
        object_centroids: List[Tuple[int, int]] = []
        object_bboxes: List[Tuple[int, int, int, int]] = []

        for object_id in object_ids:
            predicted_pos = self.predict_next_position(object_id)
            object_centroids.append(predicted_pos)
            object_bboxes.append(self.objects[object_id]["bbox"])

        detection_centroids = [det[0] for det in detections]
        detection_bboxes = [det[3] for det in detections]

        # Calculate combined distance and IoU matrix
        assignment_scores = np.zeros((len(object_centroids), len(detection_centroids)))

        for i, (obj_centroid, obj_bbox) in enumerate(
            zip(object_centroids, object_bboxes)
        ):
            for j, (det_centroid, det_bbox) in enumerate(
                zip(detection_centroids, detection_bboxes)
            ):
                distance = self.calculate_distance(obj_centroid, det_centroid)
                iou = self.calculate_iou(obj_bbox, det_bbox)

                distance_score = distance / self.max_distance
                iou_score = iou
                assignment_scores[i][j] = distance_score - iou_score

        # Assignment logic
        used_detection_indices = set()
        used_object_indices = set()
        tracked_objects = {}

        assignments = []
        for i in range(len(object_centroids)):
            for j in range(len(detection_centroids)):
                assignments.append((assignment_scores[i][j], i, j))
        assignments.sort()

        # Handle matched detections
        for score, obj_idx, det_idx in assignments:
            if obj_idx in used_object_indices or det_idx in used_detection_indices:
                continue

            object_id = object_ids[obj_idx]
            detection = detections[det_idx]
            centroid, class_id, confidence, bbox = detection

            distance = self.calculate_distance(object_centroids[obj_idx], centroid)
            iou = self.calculate_iou(object_bboxes[obj_idx], bbox)

            if distance <= self.max_distance or iou > 0.1:
                existing_class = self.object_classes[object_id]
                vehicle_classes = {2, 3, 5, 7}
                class_match = existing_class == class_id or (
                    existing_class in vehicle_classes and class_id in vehicle_classes
                )

                if class_match:
                    self.objects[object_id].update(
                        {
                            "centroid": centroid,
                            "confidence": confidence,
                            "bbox": bbox,
                            "last_seen": time.time(),
                        }
                    )
                    self.disappeared[object_id] = 0
                    self.object_history[object_id].append(centroid)

                    # IMPROVED: Update destination logic
                    if object_id in self.object_directions:
                        self.object_directions[object_id]["last_centroid"] = centroid

                        # Get current zone
                        current_zone = (
                            self.direction_manager.determine_direction_from_position(
                                centroid, self.frame_width, self.frame_height
                            )
                        )

                        origin = self.object_directions[object_id]["origin"]

                        # Update destination only if:
                        # 1. Current zone is valid
                        # 2. Current zone is different from origin
                        # 3. Object has moved significantly (confirmed moving)
                        if (
                            current_zone
                            and current_zone != origin
                            and self.objects[object_id].get("confirmed_moving", False)
                        ):

                            # Only update if destination has actually changed
                            prev_dest = self.object_directions[object_id]["destination"]
                            if prev_dest != current_zone:
                                self.object_directions[object_id][
                                    "destination"
                                ] = current_zone
                                print(
                                    f"  → Object {object_id} destination updated: {origin} → {current_zone}"
                                )

                    is_moving = self.motion_detector.is_moving(
                        object_id, centroid, self.object_history[object_id]
                    )

                    if is_moving:
                        self.objects[object_id]["confirmed_moving"] = True

                        if timestamp and object_id in self.object_timestamps:
                            self.object_timestamps[object_id][
                                "last_seen_timestamp"
                            ] = timestamp

                    if len(self.object_history[object_id]) > 10:
                        self.object_history[object_id] = self.object_history[object_id][
                            -10:
                        ]

                    tracked_objects[object_id] = self.objects[object_id]

                    used_object_indices.add(obj_idx)
                    used_detection_indices.add(det_idx)

        # Handle unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detection_indices:
                centroid, class_id, confidence, bbox = detection
                object_id = self.register(
                    centroid, class_id, confidence, bbox, timestamp
                )
                tracked_objects[object_id] = self.objects[object_id]

        # Handle unmatched existing objects
        for obj_idx in range(len(object_ids)):
            if obj_idx not in used_object_indices:
                object_id = object_ids[obj_idx]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] <= self.max_disappeared:
                    tracked_objects[object_id] = self.objects[object_id]
                else:
                    self.deregister(object_id)

        # Clean up motion detector
        self.motion_detector.cleanup_stale_candidates(list(self.objects.keys()))

        return tracked_objects


__all__ = ["EnhancedObjectTracker", "DirectionManager"]
