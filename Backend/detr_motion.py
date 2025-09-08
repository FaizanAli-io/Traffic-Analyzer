# GPU-Optimized Vehicle Detection with Motion Filtering and Timestamp Extraction
# Optimized for Ubuntu terminal usage
import cv2
import numpy as np
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from collections import defaultdict, OrderedDict
import math
from typing import Dict, List, Tuple, Optional
import time
import re
from datetime import datetime, timedelta
import pytesseract
from PIL import Image
import torchvision.transforms as transforms
import os
import gc
import psutil
# from google.colab.patches import cv2_imshow


width = None
height = None

class TimestampExtractor:
    """Extract and parse timestamps from video frames - Enhanced to extract both date and time"""

    def __init__(self, roi_height_percent=0.1, roi_width_percent=0.4):
        """
        Initialize timestamp extractor
        Args:
            roi_height_percent: Height of ROI as percentage of frame height (default: 0.1 = top 10%)
            roi_width_percent: Width of ROI as percentage of frame width (default: 0.4 = left 40%)
        """
        self.roi_height_percent = roi_height_percent
        self.roi_width_percent = roi_width_percent
        self.last_known_timestamp = None
        self.timestamp_format = "%d-%m-%Y %H:%M:%S"

    def extract_timestamp_from_frame(self, frame):
        """
        Extract timestamp from the top-left corner of a frame using OCR
        Returns both datetime object and string representation
        """
        try:
            # Get frame dimensions
            height, width = frame.shape[:2]

            # Define region of interest (top-left corner)
            roi_height = int(height * self.roi_height_percent)
            roi_width = int(width * self.roi_width_percent)

            # Extract the region of interest
            roi = frame[0:roi_height, 0:roi_width]

            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi

            # Apply threshold to make text more readable
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Increase contrast
            roi_thresh = cv2.bitwise_not(roi_thresh)

            # Use OCR to extract text - Enhanced for better date/time recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-: /'
            text = pytesseract.image_to_string(roi_thresh, config=custom_config)

            # Clean up the text
            text = text.strip().replace('\n', ' ')

            # Enhanced patterns to capture both date and time
            patterns_and_formats = [
                (r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})', "%d-%m-%Y %H:%M:%S"),  # DD-MM-YYYY HH:MM:SS
                (r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})', "%d/%m/%Y %H:%M:%S"),  # DD/MM/YYYY HH:MM:SS
                (r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', "%Y-%m-%d %H:%M:%S"),  # YYYY-MM-DD HH:MM:SS
                (r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2})', "%d-%m-%Y %H:%M"),           # DD-MM-YYYY HH:MM
                (r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})', "%d/%m/%Y %H:%M"),           # DD/MM/YYYY HH:MM
            ]

            # Try each pattern
            for pattern, date_format in patterns_and_formats:
                match = re.search(pattern, text)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        timestamp = datetime.strptime(timestamp_str, date_format)
                        self.last_known_timestamp = timestamp
                        return timestamp, timestamp_str
                    except ValueError:
                        continue

            # If no full timestamp found, try time-only pattern with last known date
            time_pattern = r'(\d{2}:\d{2}:\d{2})'
            time_match = re.search(time_pattern, text)
            if time_match and self.last_known_timestamp:
                time_str = time_match.group(1)
                # Combine with last known date
                date_str = self.last_known_timestamp.strftime("%d-%m-%Y")
                full_timestamp_str = f"{date_str} {time_str}"
                try:
                    timestamp = datetime.strptime(full_timestamp_str, "%d-%m-%Y %H:%M:%S")
                    self.last_known_timestamp = timestamp
                    return timestamp, full_timestamp_str
                except ValueError:
                    pass

            # If no pattern matches, return raw text for debugging
            raw_text = f"Raw OCR: {text}" if text else "No timestamp detected"
            return None, raw_text

        except Exception as e:
            return None, f"OCR Error: {str(e)}"

    def extract_timestamp(self, frame):
        """Extract timestamp - wrapper for compatibility with existing code"""
        timestamp_obj, timestamp_str = self.extract_timestamp_from_frame(frame)
        return timestamp_obj

class MotionDetector:
    """Detect movement to filter out stationary objects"""

    def __init__(self, movement_threshold=70, min_frames_to_confirm=5):
        self.movement_threshold = movement_threshold
        self.min_frames_to_confirm = min_frames_to_confirm
        self.candidate_objects = {}
        self.confirmed_moving_objects = set()

    def calculate_movement(self, positions):
        """Calculate total movement from position history"""
        if len(positions) < 2:
            return 0

        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            movement = math.sqrt(dx*dx + dy*dy)
            total_movement += movement

        return total_movement

    def is_moving(self, object_id, current_position, position_history):
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
                print(f"  ✓ Confirmed movement for Object ID {object_id} (total movement: {total_movement:.1f}px)")
                return True

        return False

    def cleanup_stale_candidates(self, active_object_ids):
        """Remove candidates that are no longer being tracked"""
        stale_ids = set(self.candidate_objects.keys()) - set(active_object_ids)
        for stale_id in stale_ids:
            del self.candidate_objects[stale_id]
            self.confirmed_moving_objects.discard(stale_id)

class EnhancedObjectTracker:
    """Advanced object tracker with IoU-based duplicate prevention and motion filtering"""

    def __init__(self, max_disappeared=3, max_distance=150, iou_threshold=0.6):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold

        self.object_history = defaultdict(list)
        self.object_classes = {}
        self.object_timestamps = {}
        self.object_durations = {}

        # NEW: Direction tracking
        self.direction_manager = DirectionManager()
        self.frame_width = width
        self.frame_height = height
        self.object_directions = {}  # Store origin/destination for each object

        # Add motion detector
        self.motion_detector = MotionDetector(movement_threshold=20, min_frames_to_confirm=3)


    def set_frame_dimensions(self, width, height):
        """Update frame dimensions"""
        self.frame_width = width
        self.frame_height = height


    def calculate_iou(self, box1, box2):
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

    def remove_duplicate_detections(self, detections):
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
            print(f"  Removed {len(detections) - len(filtered_detections)} duplicate detections")
        return filtered_detections

    def register(self, centroid, class_id, confidence, bbox, timestamp=None):
        """Register a new object with timestamp and origin direction"""
        # ... existing registration code remains the same ...
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'class_id': class_id,
            'confidence': confidence,
            'bbox': bbox,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'confirmed_moving': False
        }
        self.disappeared[self.next_object_id] = 0
        self.object_classes[self.next_object_id] = class_id
        self.object_history[self.next_object_id].append(centroid)

        if timestamp:
            self.object_timestamps[self.next_object_id] = {
                'first_seen_timestamp': timestamp,
                'last_seen_timestamp': timestamp
            }

        # NEW: Determine origin direction
        origin_direction = self.direction_manager.determine_direction_from_position(
            centroid, self.frame_width, self.frame_height
        )

        self.object_directions[self.next_object_id] = {
            'origin': origin_direction,
            'destination': origin_direction,
            'first_centroid': centroid,
            'last_centroid': centroid
        }

        if origin_direction:
            print(f"  → Object {self.next_object_id} entered from {origin_direction}")

        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        """Remove an object and calculate final duration + destination direction"""
        # NEW: Determine destination direction
        if object_id in self.object_directions:
            last_centroid = self.object_directions[object_id]['last_centroid']
            destination_direction = self.direction_manager.determine_direction_from_position(
                last_centroid, self.frame_width, self.frame_height
            )
            self.object_directions[object_id]['destination'] = destination_direction

            if destination_direction:
                origin = self.object_directions[object_id]['origin']
                print(f"  ← Object {object_id} exited to {destination_direction} (from {origin})")

        # ... existing deregistration code remains the same ...
        if (object_id in self.object_timestamps and
            self.objects.get(object_id, {}).get('confirmed_moving', False)):

            timestamps = self.object_timestamps[object_id]
            first_seen = timestamps['first_seen_timestamp']
            last_seen = timestamps['last_seen_timestamp']

            if first_seen and last_seen:
                duration = (last_seen - first_seen).total_seconds()
                self.object_durations[object_id] = {
                    'class': self.object_classes.get(object_id, 'unknown'),
                    'duration': duration,
                    'first_seen': first_seen,
                    'last_seen': last_seen
                }
                print(f"★ Moving Object {object_id} ({self.object_classes.get(object_id, 'unknown')}) "
                      f"completed: {duration:.2f}s visible "
                      f"({first_seen.strftime('%H:%M:%S')} → {last_seen.strftime('%H:%M:%S')})")

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

    def calculate_distance(self, point_a, point_b):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

    def predict_next_position(self, object_id):
        """Predict next position based on movement history"""
        if object_id not in self.object_history or len(self.object_history[object_id]) < 2:
            return self.objects[object_id]['centroid']

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
        # ... most of existing update code remains the same ...
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
                object_id = self.register(centroid, class_id, confidence, bbox, timestamp)
                tracked_objects[object_id] = self.objects[object_id]
            return tracked_objects

        # Enhanced matching with both distance and IoU
        object_ids = list(self.objects.keys())
        object_centroids = []
        object_bboxes = []

        for object_id in object_ids:
            predicted_pos = self.predict_next_position(object_id)
            object_centroids.append(predicted_pos)
            object_bboxes.append(self.objects[object_id]['bbox'])

        detection_centroids = [det[0] for det in detections]
        detection_bboxes = [det[3] for det in detections]

        # Calculate combined distance and IoU matrix
        assignment_scores = np.zeros((len(object_centroids), len(detection_centroids)))

        for i, (obj_centroid, obj_bbox) in enumerate(zip(object_centroids, object_bboxes)):
            for j, (det_centroid, det_bbox) in enumerate(zip(detection_centroids, detection_bboxes)):
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

        # for score, obj_idx, det_idx in assignments:
        #     if obj_idx in used_object_indices or det_idx in used_detection_indices:
        #         continue

        #     object_id = object_ids[obj_idx]
        #     detection = detections[det_idx]
        #     centroid, class_id, confidence, bbox = detection

        #     distance = self.calculate_distance(object_centroids[obj_idx], centroid)
        #     iou = self.calculate_iou(object_bboxes[obj_idx], bbox)

        #     if distance <= self.max_distance or iou > 0.1:
        #         existing_class = self.object_classes[object_id]
        #         vehicle_classes = {2, 3, 5, 7}
        #         class_match = (existing_class == class_id or
        #                      (existing_class in vehicle_classes and class_id in vehicle_classes))

        #         if class_match:
        #             self.objects[object_id].update({
        #                 'centroid': centroid,
        #                 'confidence': confidence,
        #                 'bbox': bbox,
        #                 'last_seen': time.time()
        #             })
        #             self.disappeared[object_id] = 0
        #             self.object_history[object_id].append(centroid)

        #             # NEW: Update last centroid for direction tracking
        #             if object_id in self.object_directions:
        #                 self.object_directions[object_id]['last_centroid'] = centroid

        #             is_moving = self.motion_detector.is_moving(
        #                 object_id, centroid, self.object_history[object_id]
        #             )

        #             if is_moving:
        #                 self.objects[object_id]['confirmed_moving'] = True

        #                 if timestamp and object_id in self.object_timestamps:
        #                     self.object_timestamps[object_id]['last_seen_timestamp'] = timestamp

        #             if len(self.object_history[object_id]) > 10:
        #                 self.object_history[object_id] = self.object_history[object_id][-10:]

        #             tracked_objects[object_id] = self.objects[object_id]

        #             used_object_indices.add(obj_idx)
        #             used_detection_indices.add(det_idx)
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
                class_match = (existing_class == class_id or
                            (existing_class in vehicle_classes and class_id in vehicle_classes))

                if class_match:
                    self.objects[object_id].update({
                        'centroid': centroid,
                        'confidence': confidence,
                        'bbox': bbox,
                        'last_seen': time.time()
                    })
                    self.disappeared[object_id] = 0
                    self.object_history[object_id].append(centroid)

                    # UPDATE: Continuously update destination based on current position
                    if object_id in self.object_directions:
                        self.object_directions[object_id]['last_centroid'] = centroid
                        # Update destination to current zone
                        current_destination = self.direction_manager.determine_direction_from_position(
                            centroid, self.frame_width, self.frame_height
                        )
                        self.object_directions[object_id]['destination'] = current_destination

                    is_moving = self.motion_detector.is_moving(
                        object_id, centroid, self.object_history[object_id]
                    )

                    if is_moving:
                        self.objects[object_id]['confirmed_moving'] = True

                        if timestamp and object_id in self.object_timestamps:
                            self.object_timestamps[object_id]['last_seen_timestamp'] = timestamp

                    if len(self.object_history[object_id]) > 10:
                        self.object_history[object_id] = self.object_history[object_id][-10:]

                    tracked_objects[object_id] = self.objects[object_id]

                    used_object_indices.add(obj_idx)
                    used_detection_indices.add(det_idx)

        # Handle unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detection_indices:
                centroid, class_id, confidence, bbox = detection
                object_id = self.register(centroid, class_id, confidence, bbox, timestamp)
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

# Additional classes and modifications for direction tracking

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
        print(f"Rotated counter-clockwise. New orientation: {self.get_current_mapping()}")

    def get_current_directions(self):
        """Get current direction mapping: [top, right, bottom, left]"""
        return [
            self.base_directions[self.current_orientation],                    # top
            self.base_directions[(self.current_orientation + 1) % 4],         # right
            self.base_directions[(self.current_orientation + 2) % 4],         # bottom
            self.base_directions[(self.current_orientation + 3) % 4]          # left
        ]

    def get_current_mapping(self):
        """Get readable mapping of edges to directions"""
        directions = self.get_current_directions()
        return {
            "top": directions[0],
            "right": directions[1],
            "bottom": directions[2],
            "left": directions[3]
        }

    def determine_direction_from_position(self, centroid, frame_width, frame_height, border_threshold=50):
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
            center_x, center_y = frame_width // 2, frame_height // 2
            
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
            'top': dist_to_top,
            'right': dist_to_right,
            'bottom': dist_to_bottom,
            'left': dist_to_left
        }
        
        closest_edge = min(distances.keys(), key=lambda k: distances[k])
        
        # Map edge to current direction
        current_directions = self.get_current_directions()
        edge_to_direction = {
            'top': current_directions[0],     # North
            'right': current_directions[1],   # East
            'bottom': current_directions[2],  # South
            'left': current_directions[3]     # West
        }
        
        return edge_to_direction[closest_edge]


class GPUOptimizedDETRDetector:
    """
    GPU-Optimized DETR Vehicle Detector with advanced memory management and batch processing
    """

    def __init__(self, model_name='facebook/detr-resnet-101-dc5', confidence_threshold=0.8,
                 batch_size=4, enable_mixed_precision=True):
        print("Initializing GPU-Optimized DETR Model...")
        print(f"Model: {model_name}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Batch size: {batch_size}")
        print(f"Mixed precision: {enable_mixed_precision}")

        self.video_directory = None
        self.batch_size = batch_size
        self.enable_mixed_precision = enable_mixed_precision

        # GPU configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            print("WARNING: CUDA not available, using CPU")

        # Load model with optimizations
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Enable mixed precision if requested and available
        if self.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision enabled")

        # Optimize model
        if hasattr(torch, 'jit') and torch.cuda.is_available():
            try:
                # JIT compilation for faster inference
                dummy_input = torch.randn(1, 3, 800, 800).to(self.device)
                traced_model = torch.jit.trace(self.model, {'pixel_values': dummy_input})
                self.model = traced_model
                print("Model traced with TorchScript")
            except Exception as e:
                print(f"TorchScript tracing failed: {e}, using regular model")

        self.confidence_threshold = confidence_threshold

        # Initialize tracker and other components
        self.tracker = EnhancedObjectTracker(max_disappeared=1, max_distance=100, iou_threshold=0.3)
        self.timestamp_extractor = TimestampExtractor(roi_height_percent=0.1, roi_width_percent=0.4)
        self.detection_records = []

        # COCO class mapping
        self.target_classes = {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            7: 'long-bus',
            8: 'truck'
        }

        # Colors for visualization
        self.colors = {
            1: (0, 255, 0),      # person - bright green
            2: (255, 255, 0),    # bicycle - cyan
            3: (255, 0, 0),      # car - blue
            4: (0, 165, 255),    # motorcycle - orange
            6: (0, 0, 255),      # bus - red
            7: (230, 0, 255),    # bus - dk
            8: (128, 0, 128)     # truck - purple
        }

        self.detection_stats = defaultdict(int)
        self.current_timestamp = None
        self.current_timestamp_str = None

        # Memory management
        self.frame_buffer = []
        self.clear_cache_interval = 50  # Clear cache every N frames

        print("GPU-Optimized DETR model initialized successfully!")
        print("Target classes:", list(self.target_classes.values()))

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_object_direction_info(self, object_id):
        """Get formatted origin and destination info for an object"""
        if object_id not in self.tracker.object_directions:
            return "Origin: Unknown", "Dest: Unknown"
        
        direction_info = self.tracker.object_directions[object_id]
        origin = direction_info['origin'] if direction_info['origin'] else "Unknown"
        destination = direction_info['destination'] if direction_info['destination'] else "Tracking..."
        
        return f"Origin: {origin}", f"Dest: {destination}"


    # def draw_directional_boundaries(self, frame):
    #     """Draw X-shaped directional boundaries with filled zones showing North/South/East/West regions"""
    #     height, width = frame.shape[:2]
        
    #     # Get current direction mapping
    #     current_directions = self.tracker.direction_manager.get_current_directions()
        
    #     # Define colors for each direction (semi-transparent for zones)
    #     direction_colors = {
    #         "North": (0, 255, 255),    # Yellow
    #         "South": (255, 0, 255),    # Magenta  
    #         "East": (0, 255, 0),       # Green
    #         "West": (255, 165, 0)      # Orange
    #     }
        
    #     # Create overlay for semi-transparent zones
    #     overlay = frame.copy()
        
    #     # Define X-pattern boundaries
    #     # Top-left to bottom-right diagonal
    #     diagonal1_start = (0, 0)
    #     diagonal1_end = (width, height)
        
    #     # Top-right to bottom-left diagonal  
    #     diagonal2_start = (width, 0)
    #     diagonal2_end = (0, height)
        
    #     # Create zone masks using X-pattern
    #     # North zone: above both diagonals (top triangle)
    #     north_points = np.array([[width//2, 0], [0, 0], [0, height//2], [width//2, height//2]], np.int32)
    #     north_points2 = np.array([[width//2, 0], [width, 0], [width, height//2], [width//2, height//2]], np.int32)
        
    #     # South zone: below both diagonals (bottom triangle)
    #     south_points = np.array([[width//2, height//2], [0, height//2], [0, height], [width//2, height]], np.int32)
    #     south_points2 = np.array([[width//2, height//2], [width, height//2], [width, height], [width//2, height]], np.int32)
        
    #     # East zone: right of both diagonals (right triangle)
    #     east_points = np.array([[width//2, 0], [width, 0], [width, height//2]], np.int32)
    #     east_points2 = np.array([[width//2, height], [width, height//2], [width, height]], np.int32)
        
    #     # West zone: left of both diagonals (left triangle)
    #     west_points = np.array([[0, 0], [width//2, 0], [0, height//2]], np.int32)
    #     west_points2 = np.array([[0, height//2], [width//2, height], [0, height]], np.int32)
        
    #     # Fill zones with semi-transparent colors
    #     alpha = 0.3  # Transparency level
        
    #     # North zone (top)
    #     cv2.fillPoly(overlay, [north_points], direction_colors[current_directions[0]])
    #     cv2.fillPoly(overlay, [north_points2], direction_colors[current_directions[0]])
        
    #     # South zone (bottom)  
    #     cv2.fillPoly(overlay, [south_points], direction_colors[current_directions[2]])
    #     cv2.fillPoly(overlay, [south_points2], direction_colors[current_directions[2]])
        
    #     # East zone (right)
    #     cv2.fillPoly(overlay, [east_points], direction_colors[current_directions[1]])
    #     cv2.fillPoly(overlay, [east_points2], direction_colors[current_directions[1]])
        
    #     # West zone (left)
    #     cv2.fillPoly(overlay, [west_points], direction_colors[current_directions[3]])
    #     cv2.fillPoly(overlay, [west_points2], direction_colors[current_directions[3]])
        
    #     # Blend overlay with original frame
    #     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
    #     # Draw X-pattern boundary lines (thick lines)
    #     line_thickness = 4
        
    #     # Draw the X
    #     cv2.line(frame, diagonal1_start, diagonal1_end, (255, 255, 255), line_thickness)
    #     cv2.line(frame, diagonal2_start, diagonal2_end, (255, 255, 255), line_thickness)
        
    #     # Draw border lines for detection zones
    #     border_thickness = 50
    #     detection_line_thickness = 2
        
    #     # Top detection border
    #     cv2.line(frame, (0, border_thickness), (width, border_thickness), 
    #             direction_colors[current_directions[0]], detection_line_thickness)
        
    #     # Right detection border
    #     cv2.line(frame, (width - border_thickness, 0), (width - border_thickness, height), 
    #             direction_colors[current_directions[1]], detection_line_thickness)
        
    #     # Bottom detection border
    #     cv2.line(frame, (0, height - border_thickness), (width, height - border_thickness), 
    #             direction_colors[current_directions[2]], detection_line_thickness)
        
    #     # Left detection border
    #     cv2.line(frame, (border_thickness, 0), (border_thickness, height), 
    #             direction_colors[current_directions[3]], detection_line_thickness)
        
    #     # Add direction labels with better positioning
    #     font_scale = 0.9
    #     font_thickness = 2
        
    #     # Calculate label positions
    #     # North label (top center)
    #     north_text = f"NORTH: {current_directions[0]}"
    #     (text_w, text_h), _ = cv2.getTextSize(north_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    #     cv2.putText(frame, north_text, (width//2 - text_w//2, 35),
    #               cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[0]], font_thickness)
        
    #     # South label (bottom center)
    #     south_text = f"SOUTH: {current_directions[2]}"
    #     (text_w, text_h), _ = cv2.getTextSize(south_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    #     cv2.putText(frame, south_text, (width//2 - text_w//2, height - 10),
    #               cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[2]], font_thickness)
        
    #     # East label (right center)
    #     east_text = f"EAST: {current_directions[1]}"
    #     cv2.putText(frame, east_text, (width - 180, height//2),
    #               cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[1]], font_thickness)
        
    #     # West label (left center)
    #     west_text = f"WEST: {current_directions[3]}"
    #     cv2.putText(frame, west_text, (10, height//2),
    #               cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[3]], font_thickness)
    def draw_directional_boundaries(self, frame):
        """Draw X-shaped directional boundaries without filled zones"""
        height, width = frame.shape[:2]
        
        # Get current direction mapping
        current_directions = self.tracker.direction_manager.get_current_directions()
        
        # Define colors for direction labels
        direction_colors = {
            "North": (0, 255, 255),    # Yellow
            "South": (255, 0, 255),    # Magenta  
            "East": (0, 255, 0),       # Green
            "West": (255, 165, 0)      # Orange
        }
        
        # Define X-pattern boundary lines
        # Top-left to bottom-right diagonal
        diagonal1_start = (0, 0)
        diagonal1_end = (width, height)
        
        # Top-right to bottom-left diagonal  
        diagonal2_start = (width, 0)
        diagonal2_end = (0, height)
        
        # Draw X-pattern boundary lines (thick lines)
        line_thickness = 4
        
        # Draw the X
        cv2.line(frame, diagonal1_start, diagonal1_end, (255, 255, 255), line_thickness)
        cv2.line(frame, diagonal2_start, diagonal2_end, (255, 255, 255), line_thickness)
        
        # Draw border lines for detection zones
        border_thickness = 50
        detection_line_thickness = 2
        
        # Top detection border
        cv2.line(frame, (0, border_thickness), (width, border_thickness), 
                direction_colors[current_directions[0]], detection_line_thickness)
        
        # Right detection border
        cv2.line(frame, (width - border_thickness, 0), (width - border_thickness, height), 
                direction_colors[current_directions[1]], detection_line_thickness)
        
        # Bottom detection border
        cv2.line(frame, (0, height - border_thickness), (width, height - border_thickness), 
                direction_colors[current_directions[2]], detection_line_thickness)
        
        # Left detection border
        cv2.line(frame, (border_thickness, 0), (border_thickness, height), 
                direction_colors[current_directions[3]], detection_line_thickness)
        
        # Add direction labels
        font_scale = 0.9
        font_thickness = 2
        
        # North label (top center)
        north_text = f"NORTH: {current_directions[0]}"
        (text_w, text_h), _ = cv2.getTextSize(north_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.putText(frame, north_text, (width//2 - text_w//2, 35),
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[0]], font_thickness)
        
        # South label (bottom center)
        south_text = f"SOUTH: {current_directions[2]}"
        (text_w, text_h), _ = cv2.getTextSize(south_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.putText(frame, south_text, (width//2 - text_w//2, height - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[2]], font_thickness)
        
        # East label (right center)
        east_text = f"EAST: {current_directions[1]}"
        cv2.putText(frame, east_text, (width - 180, height//2),
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[1]], font_thickness)
        
        # West label (left center)
        west_text = f"WEST: {current_directions[3]}"
        cv2.putText(frame, west_text, (10, height//2),
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, direction_colors[current_directions[3]], font_thickness)
        

    def preprocess_frames_batch(self, frames):
        """Preprocess multiple frames for batch processing"""
        processed_frames = []
        pil_images = []

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_images.append(pil_image)

        # Batch preprocessing
        try:
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            return inputs, pil_images
        except Exception as e:
            print(f"Batch preprocessing failed: {e}")
            # Fallback to single frame processing
            inputs = self.processor(images=pil_images[0], return_tensors="pt")
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            return inputs, [pil_images[0]]

    def detect_objects_batch(self, frames):
        """Detect objects using batch processing for better GPU utilization"""
        if len(frames) == 1:
            return [self.detect_objects_single(frames[0])]

        try:
            inputs, pil_images = self.preprocess_frames_batch(frames)

            with torch.no_grad():
                if self.enable_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            # Process batch results
            batch_detections = []
            target_sizes = torch.tensor([img.size[::-1] for img in pil_images]).to(self.device)

            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )

            for result in results:
                detections = []
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    class_id = label.item()
                    confidence = score.item()

                    if class_id in self.target_classes and confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        centroid_x = int((x1 + x2) / 2)
                        centroid_y = int((y1 + y2) / 2)
                        centroid = (centroid_x, centroid_y)
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        detections.append((centroid, class_id, confidence, bbox))

                batch_detections.append(detections)

            return batch_detections

        except Exception as e:
            print(f"Batch detection failed: {e}, falling back to single frame processing")
            return [self.detect_objects_single(frame) for frame in frames]

    def detect_objects_single(self, frame):
        """Detect objects in a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad():
            if self.enable_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_id = label.item()
            confidence = score.item()

            if class_id in self.target_classes and confidence >= self.confidence_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                centroid = (centroid_x, centroid_y)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                detections.append((centroid, class_id, confidence, bbox))

        return detections

    def process_frame(self, frame):
        """Process a single frame with enhanced detection and motion filtering + direction tracking"""
        # Extract timestamp
        self.current_timestamp, self.current_timestamp_str = self.timestamp_extractor.extract_timestamp_from_frame(frame)

        # Detect objects
        detections = self.detect_objects_single(frame)

        # Update tracker
        tracked_objects = self.tracker.update(detections, self.current_timestamp)

        # Record detections (MODIFIED to include origin/destination)
        for object_id, obj_info in tracked_objects.items():
            if obj_info.get('confirmed_moving', False):
                already_recorded = any(record['object_id'] == object_id for record in self.detection_records)

                if not already_recorded:
                    timestamp_str = self.current_timestamp_str if self.current_timestamp_str else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Get direction info
                    origin = None
                    destination = None
                    if object_id in self.tracker.object_directions:
                        direction_info = self.tracker.object_directions[object_id]
                        origin = direction_info['origin']
                        destination = direction_info['destination']

                    detection_record = {
                        'object_id': object_id,
                        'object_type': self.target_classes[obj_info['class_id']],
                        'timestamp': timestamp_str,
                        'origin': origin,
                        'destination': destination
                    }
                    self.detection_records.append(detection_record)

                    origin_text = origin if origin else "Unknown"
                    print(f"📝 Recorded: ID {object_id} ({self.target_classes[obj_info['class_id']]}) from {origin_text} at {timestamp_str}")

        return tracked_objects

    def draw_tracked_objects(self, frame, tracked_objects):
        """Draw enhanced annotations (only for moving objects)"""
        annotated_frame = frame.copy()

        # Draw directional boundaries first (so they appear behind objects)
        self.draw_directional_boundaries(annotated_frame)

        for object_id, obj_info in tracked_objects.items():
            if not obj_info.get('confirmed_moving', False):
                continue

            bbox = obj_info['bbox']
            class_id = obj_info['class_id']
            confidence = obj_info['confidence']
            centroid = obj_info['centroid']

            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_id, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

            # Draw centroid
            cv2.circle(annotated_frame, centroid, 6, color, -1)
            cv2.circle(annotated_frame, centroid, 6, (255, 255, 255), 2)

            # Calculate duration
            duration_text = ""
            if object_id in self.tracker.object_timestamps:
                timestamps = self.tracker.object_timestamps[object_id]
                if timestamps['first_seen_timestamp'] and self.current_timestamp:
                    current_duration = (self.current_timestamp - timestamps['first_seen_timestamp']).total_seconds()
                    duration_text = f" ({current_duration:.1f}s)"

            # Enhanced label
            # label = f"ID:{object_id} {self.target_classes[class_id]} {confidence:.2f}{duration_text} [MOVING]"
            

            # Get direction information
            origin_text, dest_text = self.get_object_direction_info(object_id)

            # Enhanced label with direction info
            label = f"ID:{object_id} {self.target_classes[class_id]} {confidence:.2f}{duration_text}"
            direction_label = f"{origin_text} -> {dest_text}"

            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Draw label background
            # cv2.rectangle(annotated_frame, (x1, y1 - text_height - 15),
            #              (x1 + text_width + 10, y1), color, -1)
            # cv2.rectangle(annotated_frame, (x1, y1 - text_height - 15),
            #              (x1 + text_width + 10, y1), (255, 255, 255), 2)

            # # Draw text
            # cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8),
            #            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Calculate text dimensions for both labels
            (main_text_width, main_text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            (dir_text_width, dir_text_height), _ = cv2.getTextSize(direction_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale-0.1, thickness-1)

            # Use wider text width for background
            max_text_width = max(main_text_width, dir_text_width)
            total_height = main_text_height + dir_text_height + 20

            # Draw label background (larger for two lines)
            cv2.rectangle(annotated_frame, (x1, y1 - total_height - 5),
                        (x1 + max_text_width + 10, y1), color, -1)
            cv2.rectangle(annotated_frame, (x1, y1 - total_height - 5),
                        (x1 + max_text_width + 10, y1), (255, 255, 255), 2)

            # Draw main label (top line)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - dir_text_height - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Draw direction label (bottom line)
            cv2.putText(annotated_frame, direction_label, (x1 + 5, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale-0.1, (0, 255, 255), thickness-1)

            # Draw movement trail
            if object_id in self.tracker.object_history:
                points = self.tracker.object_history[object_id]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        thickness = max(1, int(3 * (i / len(points))))
                        cv2.line(annotated_frame, points[i-1], points[i], color, thickness)

        return annotated_frame

    def draw_timestamp_overlay(self, frame, frame_count):
        """Draw timestamp information overlay"""
        height, width = frame.shape[:2]

        # Timestamp panel
        panel_x = width - 400
        panel_y = height - 100
        cv2.rectangle(frame, (panel_x, panel_y), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (width - 10, height - 10), (255, 255, 255), 2)

        # Display information
        timestamp_text = f"Extracted Timestamp: {self.current_timestamp_str if self.current_timestamp_str else 'None'}"
        cv2.putText(frame, timestamp_text, (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (panel_x + 10, panel_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.current_timestamp:
            video_time_text = f"Video Time: {self.current_timestamp.strftime('%H:%M:%S')}"
            cv2.putText(frame, video_time_text, (panel_x + 10, panel_y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # GPU info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_text = f"GPU Memory: {gpu_memory:.2f}GB"
            cv2.putText(frame, gpu_text, (panel_x + 10, panel_y + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

    def process_video(self, video_path, output_path=None, display=False, target_fps=30,
                     save_preview_frames=True, preview_interval=25):
        """Process video with direction tracking"""
        cap = cv2.VideoCapture(video_path)
        self.video_directory = os.path.dirname(os.path.abspath(video_path))

        if self.video_directory and not os.path.exists(self.video_directory):
            os.makedirs(self.video_directory, exist_ok=True)

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # NEW: Set frame dimensions in tracker
        self.tracker.set_frame_dimensions(width, height)

        # ... rest of process_video method remains exactly the same ...
        frame_skip = max(1, int(original_fps / target_fps))
        effective_fps = original_fps / frame_skip
        estimated_output_frames = total_frames // frame_skip

        print("="*80)
        print("GPU-OPTIMIZED DETR TRANSFORMER + MOTION DETECTION + DIRECTION TRACKING")
        print("="*80)
        print(f"Video Properties:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Original FPS: {original_fps:.2f}")
        print(f"  - Target FPS: {target_fps}")
        print(f"  - Frame Skip Ratio: {frame_skip}")
        print(f"  - Effective Processing FPS: {effective_fps:.2f}")
        print(f"  - Total Frames: {total_frames}")
        print(f"  - Direction Mapping: {self.tracker.direction_manager.get_current_mapping()}")
        print("="*80)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

        frame_count = 0
        processed_frame_count = 0
        processing_start_time = time.time()
        detection_times = []

        # Create preview directory if needed
        if save_preview_frames:
            preview_dir = os.path.join(self.video_directory, 'preview_frames')
            os.makedirs(preview_dir, exist_ok=True)
            print(f"Preview frames will be saved to: {preview_dir}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames to achieve target FPS
                if frame_count % frame_skip != 0:
                    continue

                processed_frame_count += 1
                frame_start_time = time.time()

                print(f"\n{'='*60}")
                print(f"PROCESSING FRAME {processed_frame_count} (actual frame {frame_count}/{total_frames})")

                # Process frame
                tracked_objects = self.process_frame(frame)

                frame_processing_time = time.time() - frame_start_time
                detection_times.append(frame_processing_time)

                # Print results
                raw_detections = len(self.detect_objects_single(frame))
                moving_objects = len([obj for obj in tracked_objects.values()
                                    if obj.get('confirmed_moving', False)])
                candidates = len(tracked_objects) - moving_objects

                print(f"DETR raw detections: {raw_detections}")
                print(f"Moving objects: {moving_objects}")
                print(f"Candidates (checking motion): {candidates}")
                print(f"Processing time: {frame_processing_time:.3f}s")
                print(f"Extracted timestamp: {self.current_timestamp_str}")

                if self.current_timestamp:
                    print(f"Parsed video timestamp: {self.current_timestamp.strftime('%d-%m-%Y %H:%M:%S')}")

                # Print GPU memory usage
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")

                # Print active objects
                for obj_id, obj_info in tracked_objects.items():
                    if obj_info.get('confirmed_moving', False):
                        if obj_id in self.tracker.object_timestamps:
                            timestamps = self.tracker.object_timestamps[obj_id]
                            if timestamps['first_seen_timestamp'] and self.current_timestamp:
                                current_duration = (self.current_timestamp - timestamps['first_seen_timestamp']).total_seconds()
                                class_name = self.target_classes[obj_info['class_id']]
                                confidence = obj_info['confidence']
                                print(f"  → ID {obj_id} ({class_name}): {current_duration:.1f}s | conf: {confidence:.3f} | MOVING")

                # Print candidates
                for obj_id, obj_info in tracked_objects.items():
                    if not obj_info.get('confirmed_moving', False):
                        class_name = self.target_classes[obj_info['class_id']]
                        confidence = obj_info['confidence']
                        print(f"  ? ID {obj_id} ({class_name}): conf: {confidence:.3f} | CHECKING MOTION...")

                # Draw annotations
                annotated_frame = self.draw_tracked_objects(frame, tracked_objects)
                self.draw_timestamp_overlay(annotated_frame, processed_frame_count)

                # Handle display - TERMINAL COMPATIBLE
                if display and save_preview_frames and processed_frame_count % preview_interval == 0:
                    preview_filename = os.path.join(preview_dir, f"preview_frame_{processed_frame_count:06d}.jpg")
                    cv2.imwrite(preview_filename, annotated_frame)
                    print(f"📸 Saved preview: {preview_filename}")

                # Commented out cv2.imshow for terminal compatibility
                if display:
                    print("####")
                    # cv2_imshow(annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Save frame to output video
                if out:
                    out.write(annotated_frame)

                # Memory management
                if processed_frame_count % self.clear_cache_interval == 0:
                    self.clear_gpu_memory()

                # Progress updates
                if processed_frame_count % 25 == 0 or frame_count == total_frames:
                    elapsed = time.time() - processing_start_time
                    progress = (frame_count / total_frames) * 100
                    processing_fps = processed_frame_count / elapsed if elapsed > 0 else 0
                    avg_detection_time = np.mean(detection_times)

                    print(f"\n{'*'*70}")
                    print(f"PROGRESS: {progress:.1f}% | Processing FPS: {processing_fps:.1f}")
                    print(f"Frames processed: {processed_frame_count}/{estimated_output_frames}")
                    print(f"Actual frames read: {frame_count}/{total_frames}")
                    print(f"Avg Detection Time: {avg_detection_time:.3f}s/frame")
                    print(f"Moving Objects: {len([obj for obj in self.tracker.objects.values() if obj.get('confirmed_moving', False)])}")
                    print(f"Candidates: {len([obj for obj in self.tracker.objects.values() if not obj.get('confirmed_moving', False)])}")
                    print(f"Completed Objects: {len(self.tracker.object_durations)}")
                    print(f"CSV Records: {len(self.detection_records)}")

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        print(f"GPU Memory Usage: {gpu_memory:.2f}GB")

                    print(f"System RAM: {psutil.virtual_memory().percent}%")
                    print(f"{'*'*70}")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")

        finally:
            cap.release()
            if out:
                out.release()

            # No cv2.destroyAllWindows() for terminal compatibility

            # Clear GPU memory
            self.clear_gpu_memory()

            # Final statistics
            total_time = time.time() - processing_start_time
            time_saved = (total_frames - processed_frame_count) * np.mean(detection_times) if detection_times else 0

            print(f"\n{'='*80}")
            print("PROCESSING COMPLETED")
            print(f"{'='*80}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Frames processed: {processed_frame_count}/{total_frames}")
            print(f"Average detection time: {np.mean(detection_times):.3f}s/frame")
            print(f"Processing FPS: {processed_frame_count/total_time:.2f}")
            print(f"GPU acceleration: {'Used' if torch.cuda.is_available() else 'Not available'}")
            print(f"CSV Records Created: {len(self.detection_records)}")

            # Export results
            print(f"\n{'='*60}")
            print("EXPORTING RESULTS...")
            print(f"{'='*60}")
            self.print_duration_statistics()
            self.export_detection_records()
            self.export_duration_data()

    def print_duration_statistics(self):
        """Print comprehensive duration analysis for moving objects"""
        print("\n" + "="*80)
        print("MOVING OBJECT DURATION ANALYSIS")
        print("="*80)

        if not self.tracker.object_durations:
            print("No completed moving object tracks found.")
            return

        class_durations = defaultdict(list)
        for obj_id, data in self.tracker.object_durations.items():
            class_name = data['class']
            duration = data['duration']
            class_durations[class_name].append(duration)

        print("\nSTATISTICS BY MOVING OBJECT CLASS:")
        print("-" * 60)

        total_objects = 0
        total_duration = 0

        for class_name, durations in class_durations.items():
            count = len(durations)
            avg_duration = np.mean(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            median_duration = np.median(durations)
            std_duration = np.std(durations)
            total_class_duration = sum(durations)

            total_objects += count
            total_duration += total_class_duration

            print(f"\n{class_name.upper()} ({count} moving objects):")
            print(f"  • Average Duration: {avg_duration:.2f} ± {std_duration:.2f} seconds")
            print(f"  • Median Duration: {median_duration:.2f} seconds")
            print(f"  • Range: {min_duration:.2f}s - {max_duration:.2f}s")
            print(f"  • Total Time Visible: {total_class_duration:.2f} seconds")

        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  • Total Moving Objects: {total_objects}")
        print(f"  • Total Tracking Duration: {total_duration:.2f} seconds")
        print(f"  • Average Duration per Object: {total_duration/total_objects:.2f} seconds")

        print(f"\nTRACKING TIMELINE:")
        print("-" * 60)
        for obj_id, data in sorted(self.tracker.object_durations.items()):
            first_seen = data['first_seen'].strftime('%H:%M:%S')
            last_seen = data['last_seen'].strftime('%H:%M:%S')
            duration = data['duration']
            class_name = data['class']
            print(f"ID {obj_id:2d} | {class_name:10s} | {duration:6.2f}s | {first_seen} → {last_seen}")

        print("="*80)

    def export_detection_records(self):
        """Export detection records with origin→destination summary"""
        if not self.detection_records:
            print("\n📊 No detection records to export.")
            return

        try:
            import pandas as pd

            # Create DataFrame from detection records
            df = pd.DataFrame(self.detection_records)

            # Sort by timestamp
            df = df.sort_values(['timestamp', 'object_id'])

            # Generate filename
            timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            if self.video_directory and os.path.exists(self.video_directory):
                csv_filename = os.path.join(self.video_directory, f"object_detections_with_directions_{timestamp_suffix}.csv")
            else:
                csv_filename = f"object_detections_with_directions_{timestamp_suffix}.csv"

            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_filename) if os.path.dirname(csv_filename) else '.', exist_ok=True)

            # Export to CSV
            df.to_csv(csv_filename, index=False)
            print(f"\n📊 Detection records exported to: {csv_filename}")

            # NEW: Build origin→destination summary
            summary_lines = []
            summary_lines.append("SUMMARY:")
            summary_lines.append(f"Total detection records:,{len(df)}")
            summary_lines.append(f"Object types detected:,{', '.join(df['object_type'].unique())}")

            if 'timestamp' in df.columns and len(df) > 0:
                summary_lines.append(f"Time range:,{df['timestamp'].min()} to {df['timestamp'].max()}")

            summary_lines.append("Detection counts by type:")
            type_counts = df['object_type'].value_counts()
            for obj_type, count in type_counts.items():
                summary_lines.append(f"{obj_type},{count}")

            # NEW: Origin→Destination Summary
            summary_lines.append("\nORIGIN→DESTINATION SUMMARY:")

            # Filter out rows with unknown origin or destination
            direction_df = df[(df['origin'].notna()) & (df['destination'].notna()) &
                            (df['origin'] != 'None') & (df['destination'] != 'None')]

            if len(direction_df) > 0:
                direction_counts = direction_df.groupby(['origin', 'destination']).size().reset_index(name='count')

                for _, row in direction_counts.iterrows():
                    origin = row['origin']
                    destination = row['destination']
                    count = row['count']
                    summary_lines.append(f"{origin}->{destination},{count}")

                total_with_directions = len(direction_df)
                summary_lines.append(f"Total with known directions:,{total_with_directions}")
            else:
                summary_lines.append("No complete origin->destination pairs found,0")

            # Append summary to the same CSV
            with open(csv_filename, "a", encoding="utf-8") as f:
                f.write("\n\n\n")  # leave blank lines
                for line in summary_lines:
                    f.write(line + "\n")

            # Print summary
            print("\n" + "\n".join(["  • " + l.replace(",", " ") for l in summary_lines[1:]]))

        except ImportError:
            print("\n📝 pandas not available, using manual CSV export...")
            self.export_detection_records_manual()
        except Exception as e:
            print(f"\n⚠ CSV export failed: {e}")
            self.export_detection_records_manual()


    # NEW: Add rotation control methods
    def rotate_directions_clockwise(self):
        """Rotate direction mapping clockwise"""
        self.tracker.direction_manager.rotate_clockwise()

    def rotate_directions_counterclockwise(self):
        """Rotate direction mapping counter-clockwise"""
        self.tracker.direction_manager.rotate_counterclockwise()

    def get_current_direction_mapping(self):
        """Get current direction mapping"""
        return self.tracker.direction_manager.get_current_mapping()


    def export_detection_records_manual(self):
        """Manual CSV export without pandas"""
        if not self.detection_records:
            return

        try:
            timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = os.path.join(self.video_directory, f"object_detections_{timestamp_suffix}.csv")

            with open(csv_filename, 'w', newline='') as csvfile:
                if self.detection_records:
                    headers = list(self.detection_records[0].keys())
                    csvfile.write(','.join(headers) + '\n')

                    for record in self.detection_records:
                        row = [str(record[header]) for header in headers]
                        csvfile.write(','.join(row) + '\n')

            print(f"\n📊 Detection records exported to: {csv_filename}")
            print(f"  • Total records: {len(self.detection_records)}")

        except Exception as e:
            print(f"\n⚠ Manual CSV export failed: {e}")

    def export_duration_data(self):
        """Export duration data to CSV"""
        if not self.tracker.object_durations:
            print("\n📊 No duration data to export.")
            return

        try:
            import pandas as pd

            export_data = []
            for obj_id, data in self.tracker.object_durations.items():
                export_data.append({
                    'object_id': obj_id,
                    'class': data['class'],
                    'duration_seconds': data['duration'],
                    'first_seen': data['first_seen'].strftime('%Y-%m-%d %H:%M:%S'),
                    'last_seen': data['last_seen'].strftime('%Y-%m-%d %H:%M:%S'),
                    'movement_confirmed': True
                })

            if export_data:
                df = pd.DataFrame(export_data)
                df = df.sort_values(['class', 'duration_seconds'], ascending=[True, False])

                timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_filename = os.path.join(self.video_directory, f"moving_vehicle_durations_{timestamp_suffix}.csv")

                df.to_csv(csv_filename, index=False)
                print(f"\n📊 Duration data exported to: {csv_filename}")

        except ImportError:
            self.export_duration_data_manual()
        except Exception as e:
            print(f"\n⚠ Duration CSV export failed: {e}")

    def export_duration_data_manual(self):
        """Manual CSV export for duration data"""
        if not self.tracker.object_durations:
            return

        try:
            timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = os.path.join(self.video_directory, f"moving_vehicle_durations_{timestamp_suffix}.csv")

            with open(csv_filename, 'w', newline='') as csvfile:
                headers = ['object_id', 'class', 'duration_seconds', 'first_seen', 'last_seen', 'movement_confirmed']
                csvfile.write(','.join(headers) + '\n')

                for obj_id, data in self.tracker.object_durations.items():
                    row = [
                        str(obj_id),
                        data['class'],
                        str(data['duration']),
                        data['first_seen'].strftime('%Y-%m-%d %H:%M:%S'),
                        data['last_seen'].strftime('%Y-%m-%d %H:%M:%S'),
                        'True'
                    ]
                    csvfile.write(','.join(row) + '\n')

            print(f"\n📊 Duration data exported to: {csv_filename}")

        except Exception as e:
            print(f"\n⚠ Manual duration CSV export failed: {e}")

# Add this import at the top of your detr_motion.py file (after existing imports)
import argparse

# Modify the main() function to accept command line arguments
def main():
    """
    Main function for GPU-optimized DETR detection - Terminal Compatible
    """
    # Add argument parsing
    parser = argparse.ArgumentParser(description='GPU-Optimized DETR Vehicle Detection')
    parser.add_argument('--direction-orientation', type=int, default=0, 
                       help='Direction orientation (0=North up, 1=East up, 2=South up, 3=West up)')
    args = parser.parse_args()
    
    print("="*80)
    print("GPU-OPTIMIZED DETR VEHICLE DETECTION - TERMINAL VERSION")
    print("="*80)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name()}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ GPU not available - using CPU (will be slower)")

    # Initialize detector with GPU optimizations
    detector = GPUOptimizedDETRDetector(
        model_name='facebook/detr-resnet-101-dc5',  # High accuracy model
        confidence_threshold=0.75,
        batch_size=4,  # Adjust based on your GPU memory
        enable_mixed_precision=True  # Faster inference on modern GPUs
    )

    # NEW: Set the direction orientation from command line argument
    detector.tracker.direction_manager.current_orientation = args.direction_orientation
    print(f"Direction orientation set to: {args.direction_orientation}")
    print(f"Direction mapping: {detector.tracker.direction_manager.get_current_mapping()}")

    # Configure paths
    video_path = "input_video_4.mp4"  # Updated to match your fixed filename
    output_path = "output_detr_motion_filtered.mp4"

    try:
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display=False,  # Set to False for terminal usage
            target_fps=10,  # Adjusted for better performance
            save_preview_frames=True,  # Save preview frames instead of displaying
            preview_interval=25  # Save every 25th frame as preview
        )
    except FileNotFoundError:
        print(f"❌ Video file '{video_path}' not found!")
        print("Please update the video_path variable with the correct path.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()