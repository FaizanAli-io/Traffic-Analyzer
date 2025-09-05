#bilal code 8/30/2025 - Enhanced with Motion Detection (Ubuntu Terminal Version)
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

class TimestampExtractor:
    """Extract and parse timestamps from video frames"""

    def __init__(self, timestamp_region=(0, 0, 300, 50)):
        self.timestamp_region = timestamp_region
        self.last_known_timestamp = None
        self.timestamp_format = "%d-%m-%Y %H:%M:%S"

    def extract_timestamp(self, frame):
        """Extract timestamp from frame"""
        try:
            x, y, w, h = self.timestamp_region
            timestamp_roi = frame[y:y+h, x:x+w]

            gray = cv2.cvtColor(timestamp_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)

            text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=0123456789-: ')
            text = text.strip()

            timestamp_pattern = r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})'
            match = re.search(timestamp_pattern, text)

            if match:
                timestamp_str = match.group(1)
                timestamp = datetime.strptime(timestamp_str, self.timestamp_format)
                self.last_known_timestamp = timestamp
                return timestamp
            else:
                return self.estimate_timestamp()

        except Exception as e:
            return self.estimate_timestamp()

    def estimate_timestamp(self, fps=25):
        """Estimate timestamp if extraction fails"""
        if self.last_known_timestamp:
            frame_duration = timedelta(seconds=1/fps)
            estimated = self.last_known_timestamp + frame_duration
            self.last_known_timestamp = estimated
            return estimated
        return None

class MotionDetector:
    """Detect movement to filter out stationary objects"""

    def __init__(self, movement_threshold=70, min_frames_to_confirm=5):
        self.movement_threshold = movement_threshold  # Minimum pixel movement to consider motion
        self.min_frames_to_confirm = min_frames_to_confirm  # Frames needed to confirm movement
        self.candidate_objects = {}  # Store potential objects before confirming movement
        self.confirmed_moving_objects = set()  # IDs of objects confirmed to be moving

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
        # If already confirmed as moving, continue tracking
        if object_id in self.confirmed_moving_objects:
            return True

        # Add current position to history
        if object_id not in self.candidate_objects:
            self.candidate_objects[object_id] = [current_position]
            return False

        self.candidate_objects[object_id].append(current_position)

        # Keep only recent positions
        if len(self.candidate_objects[object_id]) > 10:
            self.candidate_objects[object_id] = self.candidate_objects[object_id][-10:]

        # Check if we have enough frames to analyze movement
        if len(self.candidate_objects[object_id]) >= self.min_frames_to_confirm:
            total_movement = self.calculate_movement(self.candidate_objects[object_id])

            # If movement exceeds threshold, confirm as moving object
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
    """
    Advanced object tracker with IoU-based duplicate prevention and motion filtering
    """

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

        # Add motion detector
        self.motion_detector = MotionDetector(movement_threshold=20, min_frames_to_confirm=3)

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
        detections_with_idx.sort(key=lambda x: x[1][2], reverse=True)  # Sort by confidence

        filtered_detections = []
        used_indices = set()

        for i, (orig_idx, detection) in enumerate(detections_with_idx):
            if orig_idx in used_indices:
                continue

            centroid, class_id, confidence, bbox = detection
            is_duplicate = False

            # Check against already filtered detections
            for existing_detection in filtered_detections:
                existing_bbox = existing_detection[3]
                existing_class = existing_detection[1]

                # Only check IoU for same class objects
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
        """Register a new object with timestamp (but don't confirm until movement detected)"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'class_id': class_id,
            'confidence': confidence,
            'bbox': bbox,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'confirmed_moving': False  # New flag for movement confirmation
        }
        self.disappeared[self.next_object_id] = 0
        self.object_classes[self.next_object_id] = class_id
        self.object_history[self.next_object_id].append(centroid)

        if timestamp:
            self.object_timestamps[self.next_object_id] = {
                'first_seen_timestamp': timestamp,
                'last_seen_timestamp': timestamp
            }

        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        """Remove an object and calculate final duration (only for confirmed moving objects)"""
        # Only record duration for confirmed moving objects
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
        # Remove duplicates first
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

                # Combined score: lower distance is better, higher IoU is better
                distance_score = distance / self.max_distance  # 0-1 (lower is better)
                iou_score = iou  # 0-1 (higher is better)

                # Combined score (lower is better for assignment)
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

        for score, obj_idx, det_idx in assignments:
            if obj_idx in used_object_indices or det_idx in used_detection_indices:
                continue

            object_id = object_ids[obj_idx]
            detection = detections[det_idx]
            centroid, class_id, confidence, bbox = detection

            # Check distance and IoU thresholds
            distance = self.calculate_distance(object_centroids[obj_idx], centroid)
            iou = self.calculate_iou(object_bboxes[obj_idx], bbox)

            if distance <= self.max_distance or iou > 0.1:
                # Class consistency check
                existing_class = self.object_classes[object_id]
                vehicle_classes = {2, 3, 5, 7}
                class_match = (existing_class == class_id or
                             (existing_class in vehicle_classes and class_id in vehicle_classes))

                if class_match:
                    # Update existing object
                    self.objects[object_id].update({
                        'centroid': centroid,
                        'confidence': confidence,
                        'bbox': bbox,
                        'last_seen': time.time()
                    })
                    self.disappeared[object_id] = 0
                    self.object_history[object_id].append(centroid)

                    # Check for movement using motion detector
                    is_moving = self.motion_detector.is_moving(
                        object_id, centroid, self.object_history[object_id]
                    )

                    # Only mark as confirmed moving if motion is detected
                    if is_moving:
                        self.objects[object_id]['confirmed_moving'] = True

                        # Update timestamp only for moving objects
                        if timestamp and object_id in self.object_timestamps:
                            self.object_timestamps[object_id]['last_seen_timestamp'] = timestamp

                    if len(self.object_history[object_id]) > 10:
                        self.object_history[object_id] = self.object_history[object_id][-10:]

                    # Only add to tracked_objects if confirmed moving
                    if self.objects[object_id].get('confirmed_moving', False):
                        tracked_objects[object_id] = self.objects[object_id]

                    used_object_indices.add(obj_idx)
                    used_detection_indices.add(det_idx)

        # Handle unmatched detections (potential new objects)
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detection_indices:
                centroid, class_id, confidence, bbox = detection
                object_id = self.register(centroid, class_id, confidence, bbox, timestamp)
                # Don't add to tracked_objects yet - wait for movement confirmation

        # Handle unmatched existing objects
        for obj_idx in range(len(object_ids)):
            if obj_idx not in used_object_indices:
                object_id = object_ids[obj_idx]
                self.disappeared[object_id] += 1

                # Only keep tracking if confirmed moving and not disappeared too long
                if (self.disappeared[object_id] <= self.max_disappeared and
                    self.objects[object_id].get('confirmed_moving', False)):
                    tracked_objects[object_id] = self.objects[object_id]
                else:
                    self.deregister(object_id)

        # Clean up motion detector
        self.motion_detector.cleanup_stale_candidates(list(self.objects.keys()))

        return tracked_objects

class DETRVehicleDetector:
    """
    DETR (Detection Transformer) based vehicle and pedestrian detector with motion filtering
    Only tracks objects that are actually moving to avoid false positives on static objects
    """

    def __init__(self, model_name='facebook/detr-resnet-101-dc5', confidence_threshold=0.8):
        print("Initializing DETR Transformer Model with Motion Detection...")
        print(f"Model: {model_name}")
        print(f"Confidence threshold: {confidence_threshold}")

        # Load DETR model and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold

        # Initialize enhanced tracker with motion detection
        self.tracker = EnhancedObjectTracker(max_disappeared=1, max_distance=100, iou_threshold=0.3)
        self.timestamp_extractor = TimestampExtractor()

        # COCO class mapping (DETR uses COCO classes)
        self.target_classes = {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            7: 'long-bus',
            8: 'truck'
        }

        # Enhanced colors for better visibility
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

        print("DETR model with motion detection initialized successfully!")
        print("Target classes:", list(self.target_classes.values()))
        print("Motion filtering: Only objects that move will be tracked")

    def preprocess_frame(self, frame):
        """Preprocess frame for DETR model"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Use DETR processor for preprocessing
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs, pil_image

    def detect_objects(self, frame):
        """Detect objects using DETR transformer model"""
        inputs, pil_image = self.preprocess_frame(frame)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to detections
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]

        detections = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_id = label.item()
            confidence = score.item()

            # Filter by target classes and confidence
            if class_id in self.target_classes and confidence >= self.confidence_threshold:
                # Convert box format (DETR returns [x_min, y_min, x_max, y_max])
                x1, y1, x2, y2 = box.cpu().numpy()

                # Calculate centroid
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                centroid = (centroid_x, centroid_y)

                bbox = (int(x1), int(y1), int(x2), int(y2))
                detections.append((centroid, class_id, confidence, bbox))

        return detections

    def process_frame(self, frame):
        """Process a single frame with enhanced detection and motion filtering"""
        # Extract timestamp
        self.current_timestamp = self.timestamp_extractor.extract_timestamp(frame)

        # Detect objects using DETR
        detections = self.detect_objects(frame)

        # Update tracker with enhanced matching and motion filtering
        tracked_objects = self.tracker.update(detections, self.current_timestamp)

        return tracked_objects

    def draw_tracked_objects(self, frame, tracked_objects):
        """Draw enhanced annotations with duration and confidence (only for moving objects)"""
        annotated_frame = frame.copy()

        for object_id, obj_info in tracked_objects.items():
            # Only draw confirmed moving objects
            if not obj_info.get('confirmed_moving', False):
                continue

            bbox = obj_info['bbox']
            class_id = obj_info['class_id']
            confidence = obj_info['confidence']
            centroid = obj_info['centroid']

            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_id, (255, 255, 255))

            # Draw thicker bounding box for better visibility
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

            # Draw centroid
            cv2.circle(annotated_frame, centroid, 6, color, -1)
            cv2.circle(annotated_frame, centroid, 6, (255, 255, 255), 2)

            # Calculate current duration
            duration_text = ""
            if object_id in self.tracker.object_timestamps:
                timestamps = self.tracker.object_timestamps[object_id]
                if timestamps['first_seen_timestamp'] and self.current_timestamp:
                    current_duration = (self.current_timestamp - timestamps['first_seen_timestamp']).total_seconds()
                    duration_text = f" ({current_duration:.1f}s)"

            # Enhanced label with duration and movement indicator
            label = f"ID:{object_id} {self.target_classes[class_id]} {confidence:.2f}{duration_text} [MOVING]"

            # Multi-line label for better readability
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 15),
                         (x1 + text_width + 10, y1), color, -1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 15),
                         (x1 + text_width + 10, y1), (255, 255, 255), 2)

            # Draw text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Draw movement trail for confirmed moving objects
            if object_id in self.tracker.object_history:
                points = self.tracker.object_history[object_id]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        thickness = max(1, int(3 * (i / len(points))))  # Fade effect
                        cv2.line(annotated_frame, points[i-1], points[i], color, thickness)

        return annotated_frame

    def draw_statistics(self, frame, tracked_objects, frame_count):
        """Draw enhanced statistics overlay with motion detection info"""
        current_counts = defaultdict(int)
        candidate_counts = defaultdict(int)

        # Count confirmed moving objects
        for obj_info in tracked_objects.values():
            if obj_info.get('confirmed_moving', False):
                class_name = self.target_classes[obj_info['class_id']]
                current_counts[class_name] += 1

        # Count candidate objects (detected but not yet confirmed moving)
        total_candidates = 0
        for obj_id, obj_info in self.tracker.objects.items():
            if not obj_info.get('confirmed_moving', False):
                class_name = self.target_classes[obj_info['class_id']]
                candidate_counts[class_name] += 1
                total_candidates += 1

        # Larger statistics panel
        panel_height = 280
        cv2.rectangle(frame, (10, 10), (500, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, panel_height), (255, 255, 255), 2)

        # Header
        cv2.putText(frame, "DETR + MOTION DETECTION", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Frame and timestamp info
        cv2.putText(frame, f"Frame: {frame_count}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.current_timestamp:
            timestamp_str = self.current_timestamp.strftime("%d-%m-%Y %H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp_str}", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Moving objects section
        y_offset = 105
        cv2.putText(frame, "Moving Objects:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset += 25
        if current_counts:
            for class_name, count in current_counts.items():
                cv2.putText(frame, f"• {class_name}: {count}", (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
        else:
            cv2.putText(frame, "• No moving objects detected", (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            y_offset += 20

        # Candidate objects section
        cv2.putText(frame, "Candidates (checking motion):", (20, y_offset + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        y_offset += 35
        if candidate_counts:
            for class_name, count in candidate_counts.items():
                cv2.putText(frame, f"• {class_name}: {count}", (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
        else:
            cv2.putText(frame, "• No candidates", (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            y_offset += 20

        # Summary statistics
        total_active_moving = len([obj for obj in tracked_objects.values()
                                 if obj.get('confirmed_moving', False)])
        total_completed = len(self.tracker.object_durations)

        cv2.putText(frame, f"Moving: {total_active_moving} | Candidates: {total_candidates} | Done: {total_completed}", (20, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def process_video(self, video_path, output_path=None, display=True, target_fps=30):
        """Process video with DETR transformer detection at target FPS and motion filtering"""
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame skip ratio to achieve target FPS
        frame_skip = max(1, int(original_fps / target_fps))
        effective_fps = original_fps / frame_skip
        estimated_output_frames = total_frames // frame_skip

        print("="*80)
        print("DETR TRANSFORMER + MOTION DETECTION")
        print("="*80)
        print(f"Video Properties:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Original FPS: {original_fps:.2f}")
        print(f"  - Target FPS: {target_fps}")
        print(f"  - Frame Skip Ratio: {frame_skip} (processing every {frame_skip} frame(s))")
        print(f"  - Effective Processing FPS: {effective_fps:.2f}")
        print(f"  - Total Frames: {total_frames}")
        print(f"  - Frames to Process: ~{estimated_output_frames}")
        print(f"  - Estimated Duration: {total_frames/original_fps:.2f} seconds")
        print(f"  - Motion Detection: Only moving objects will be tracked")
        print(f"  - Static Object Filtering: Enabled")
        print("="*80)

        # Setup video writer with target FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

        frame_count = 0
        processed_frame_count = 0
        processing_start_time = time.time()
        detection_times = []

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

                # Process frame with DETR and motion detection
                tracked_objects = self.process_frame(frame)

                frame_processing_time = time.time() - frame_start_time
                detection_times.append(frame_processing_time)

                # Print frame results
                raw_detections = len(self.detect_objects(frame))
                moving_objects = len([obj for obj in tracked_objects.values()
                                    if obj.get('confirmed_moving', False)])
                candidates = len(self.tracker.objects) - moving_objects

                print(f"DETR raw detections: {raw_detections}")
                print(f"Moving objects: {moving_objects}")
                print(f"Candidates (checking motion): {candidates}")
                print(f"Processing time: {frame_processing_time:.3f}s")

                if self.current_timestamp:
                    print(f"Video timestamp: {self.current_timestamp.strftime('%H:%M:%S')}")

                # Print active moving objects with durations
                moving_count = 0
                for obj_id, obj_info in tracked_objects.items():
                    if obj_info.get('confirmed_moving', False):
                        moving_count += 1
                        if obj_id in self.tracker.object_timestamps:
                            timestamps = self.tracker.object_timestamps[obj_id]
                            if timestamps['first_seen_timestamp'] and self.current_timestamp:
                                current_duration = (self.current_timestamp - timestamps['first_seen_timestamp']).total_seconds()
                                class_name = self.target_classes[obj_info['class_id']]
                                confidence = obj_info['confidence']
                                print(f"  → ID {obj_id} ({class_name}): {current_duration:.1f}s | conf: {confidence:.3f} | MOVING")

                # Print candidate objects (detected but motion not confirmed)
                candidate_count = 0
                for obj_id, obj_info in self.tracker.objects.items():
                    if not obj_info.get('confirmed_moving', False):
                        candidate_count += 1
                        class_name = self.target_classes[obj_info['class_id']]
                        confidence = obj_info['confidence']
                        print(f"  ? ID {obj_id} ({class_name}): conf: {confidence:.3f} | CHECKING MOTION...")

                # Draw enhanced annotations (only moving objects will be visible)
                annotated_frame = self.draw_tracked_objects(frame, tracked_objects)
                self.draw_statistics(annotated_frame, tracked_objects, processed_frame_count)

                # Save frame
                if out:
                    out.write(annotated_frame)

                # Optional display for debugging (disabled by default for headless systems)
                if display:
                    cv2.imshow('DETR Motion Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped by user")
                        break

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
                    print(f"Time savings: {((total_frames - processed_frame_count) * avg_detection_time):.1f}s")
                    print(f"{'*'*70}")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            # Final statistics
            total_time = time.time() - processing_start_time
            time_saved = (total_frames - processed_frame_count) * np.mean(detection_times) if detection_times else 0

            print(f"\n{'='*80}")
            print("PROCESSING COMPLETED")
            print(f"{'='*80}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Frames processed: {processed_frame_count}/{total_frames}")
            print(f"Frame skip ratio: {frame_skip}")
            print(f"Average detection time: {np.mean(detection_times):.3f}s/frame")
            print(f"Processing FPS: {processed_frame_count/total_time:.2f}")
            print(f"Estimated time saved: {time_saved:.2f}s")
            print(f"Speedup factor: {frame_skip:.1f}x")

            self.print_duration_statistics()

    def print_duration_statistics(self):
        """Print comprehensive duration analysis for moving objects only"""
        print("\n" + "="*80)
        print("MOVING OBJECT DURATION ANALYSIS")
        print("="*80)

        if not self.tracker.object_durations:
            print("No completed moving object tracks found.")
            print("Note: Only objects confirmed to be moving are included in duration statistics.")
            return

        # Detailed analysis by class
        class_durations = defaultdict(list)
        for obj_id, data in self.tracker.object_durations.items():
            class_name = data['class']
            duration = data['duration']
            class_durations[class_name].append(duration)

        print("\nDETAILED STATISTICS BY MOVING OBJECT CLASS:")
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
            print(f"  • % of Total Tracking: {(total_class_duration/sum([sum(d) for d in class_durations.values()])*100):.1f}%")

        print(f"\n{'='*60}")
        print(f"MOVING OBJECTS SUMMARY:")
        print(f"  • Total Moving Objects Tracked: {total_objects}")
        print(f"  • Total Tracking Duration: {total_duration:.2f} seconds")
        print(f"  • Average Duration per Moving Object: {total_duration/total_objects:.2f} seconds")

        # Find longest and shortest tracked moving objects
        all_durations = [(obj_id, data['duration'], data['class'])
                        for obj_id, data in self.tracker.object_durations.items()]
        all_durations.sort(key=lambda x: x[1])

        if all_durations:
            shortest = all_durations[0]
            longest = all_durations[-1]
            print(f"  • Shortest Moving Track: ID {shortest[0]} ({shortest[2]}) - {shortest[1]:.2f}s")
            print(f"  • Longest Moving Track: ID {longest[0]} ({longest[2]}) - {longest[1]:.2f}s")

        print(f"\nMOVING OBJECT TRACKING TIMELINE:")
        print("-" * 60)
        for obj_id, data in sorted(self.tracker.object_durations.items()):
            first_seen = data['first_seen'].strftime('%H:%M:%S')
            last_seen = data['last_seen'].strftime('%H:%M:%S')
            duration = data['duration']
            class_name = data['class']

            print(f"ID {obj_id:2d} | {class_name:10s} | {duration:6.2f}s | {first_seen} → {last_seen} | CONFIRMED MOVING")

        print("="*80)

        # Export duration data to CSV for further analysis
        self.export_duration_data()

    def export_duration_data(self):
        """Export duration data to CSV file for moving objects only"""
        try:
            import pandas as pd

            # Prepare data for export
            export_data = []
            for obj_id, data in self.tracker.object_durations.items():
                export_data.append({
                    'object_id': obj_id,
                    'class': data['class'],
                    'duration_seconds': data['duration'],
                    'first_seen': data['first_seen'].strftime('%d-%m-%Y %H:%M:%S'),
                    'last_seen': data['last_seen'].strftime('%d-%m-%Y %H:%M:%S'),
                    'first_seen_time_only': data['first_seen'].strftime('%H:%M:%S'),
                    'last_seen_time_only': data['last_seen'].strftime('%H:%M:%S'),
                    'movement_confirmed': True  # All exported objects are confirmed moving
                })

            if export_data:
                df = pd.DataFrame(export_data)
                df = df.sort_values(['class', 'duration_seconds'], ascending=[True, False])

                # Save to CSV
                csv_filename = f"moving_vehicle_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_filename, index=False)
                print(f"\n📊 Moving object data exported to: {csv_filename}")

                # Print summary statistics
                print(f"\nCSV EXPORT SUMMARY:")
                print(f"  • Total moving object records: {len(df)}")
                print(f"  • Moving classes detected: {', '.join(df['class'].unique())}")
                print(f"  • Time range: {df['first_seen'].min()} to {df['last_seen'].max()}")
                print(f"  • Static objects filtered out: Motion detection prevented false positives")

        except ImportError:
            print("\n📝 Note: Install pandas to enable CSV export (pip install pandas)")
        except Exception as e:
            print(f"\n⚠️ CSV export failed: {e}")

def main():
    """
    Main function with setup instructions for DETR transformer with motion detection
    """
    print("="*80)
    print("DETR TRANSFORMER + MOTION DETECTION SETUP")
    print("="*80)
    print("\nREQUIRED INSTALLATIONS:")
    print("pip install transformers torch torchvision")
    print("pip install pytesseract pandas opencv-python pillow")
    print("sudo apt-get install tesseract-ocr  # For Ubuntu/Debian")
    print("\nOPTIONAL GPU SETUP:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("="*80)

    print("\nMOTION DETECTION FEATURES:")
    print("✓ Only tracks objects that actually move")
    print("✓ Filters out static objects (gutters, benches, etc.)")
    print("✓ Requires minimum movement over multiple frames")
    print("✓ Prevents false positives on stationary objects")
    print("✓ Stops tracking when objects leave frame")
    print("="*80)

    # Initialize DETR detector with motion detection
    detector = DETRVehicleDetector(
        model_name='facebook/detr-resnet-101-dc5',  # Can also use 'facebook/detr-resnet-101' for better accuracy
        confidence_threshold=0.75  # You can keep this at 0.7 since motion filtering handles false positives
    )

    video_path = "input_video_4.mp4"  # Replace with your video path
    output_path = "output_detr_motion_filtered.mp4"

    print("\n🚗 DETR TRANSFORMER + MOTION DETECTION")
    print("Key Improvements:")
    print("  ✓ Motion verification prevents static object detection")
    print("  ✓ Gutters, benches, poles won't be marked as cars")
    print("  ✓ Only moving vehicles/objects are tracked")
    print("  ✓ Objects must show consistent movement to be confirmed")
    print("  ✓ Automatic cleanup when objects leave frame")
    print("  ✓ Enhanced duplicate prevention")
    print("  ✓ Timestamp-based duration calculation")
    print("\nControls:")
    print("  - Press 'q' to quit (if display is enabled)")
    print("  - Only moving object data will be exported to CSV")
    print("  - Run with display=False for headless systems")
    print()

    try:
        # For headless Ubuntu systems, set display=False
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display=False,  # Set to True if you have GUI access
            target_fps=10  # Adjust as needed
        )
    except FileNotFoundError:
        print(f"❌ Video file '{video_path}' not found!")
        print("Please update the video_path variable with the correct path.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure all required packages are installed")
        print("2. Check if GPU/CUDA is properly configured")
        print("3. Verify video file path and format")
        print("4. Motion detection may need tuning for your specific video")
        print("5. For headless systems, ensure display=False in process_video()")

if __name__ == "__main__":
    main()