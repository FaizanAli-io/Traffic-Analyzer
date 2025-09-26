"""GPU DETR Detector implementation (moved from detr_motion.py)."""

from __future__ import annotations

import gc
import os
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
import psutil
import time
import os

from .ocr import TimestampExtractor
from .tracking import EnhancedObjectTracker


class GPUOptimizedDETRDetector:
    """
    GPU-Optimized DETR Vehicle Detector with advanced memory management and batch processing
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-101-dc5",
        confidence_threshold: float = 0.8,
        batch_size: int = 4,
        enable_mixed_precision: bool = True,
    ):
        print("Initializing GPU-Optimized DETR Model...")
        print(f"Model: {model_name}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Batch size: {batch_size}")
        print(f"Mixed precision: {enable_mixed_precision}")

        self.video_directory = None
        self.batch_size = batch_size
        self.enable_mixed_precision = enable_mixed_precision

        # GPU configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
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
        if hasattr(torch, "jit") and torch.cuda.is_available():
            try:
                # JIT compilation for faster inference
                dummy_input = torch.randn(1, 3, 800, 800).to(self.device)
                traced_model = torch.jit.trace(
                    self.model, {"pixel_values": dummy_input}
                )
                self.model = traced_model
                print("Model traced with TorchScript")
            except Exception as e:
                print(f"TorchScript tracing failed: {e}, using regular model")

        self.confidence_threshold = confidence_threshold

        # Initialize tracker and other components
        self.tracker = EnhancedObjectTracker(
            max_disappeared=1, max_distance=100, iou_threshold=0.3
        )
        self.timestamp_extractor = TimestampExtractor(
            roi_height_percent=0.1, roi_width_percent=0.4
        )
        self.detection_records = {}

        # COCO class mapping
        self.target_classes = {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            6: "bus",
            7: "long-bus",
            8: "truck",
        }

        # Colors for visualization
        self.colors = {
            1: (0, 255, 0),  # person - bright green
            2: (255, 255, 0),  # bicycle - cyan
            3: (255, 0, 0),  # car - blue
            4: (0, 165, 255),  # motorcycle - orange
            6: (0, 0, 255),  # bus - red
            7: (230, 0, 255),  # bus - dk
            8: (128, 0, 128),  # truck - purple
        }

        self.detection_stats = defaultdict(int)
        self.current_timestamp = None
        self.current_timestamp_str = None

        # Memory management
        self.frame_buffer: List = []
        self.clear_cache_interval = 50  # Clear cache every N frames

        print("GPU-Optimized DETR model initialized successfully!")
        print("Target classes:", list(self.target_classes.values()))

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # --- The methods below are identical to the original implementation ---
    def update_detection_record(self, object_id, obj_info, current_zone, timestamp):
        """Update or create detection record for an object"""

        if object_id not in self.detection_records:
            # Create new record - origin and destination start the same
            self.detection_records[object_id] = {
                "object_id": object_id,
                "object_type": self.target_classes[obj_info["class_id"]],
                "first_timestamp": (
                    timestamp
                    if timestamp
                    else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
                "last_timestamp": (
                    timestamp
                    if timestamp
                    else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
                "origin": current_zone,
                "destination": current_zone,  # Initially same as origin
                "zone_history": [current_zone],  # Track zone progression
                "confirmed_moving": obj_info.get("confirmed_moving", False),
            }
            print(
                f"📝 Created record for ID {object_id} ({self.target_classes[obj_info['class_id']]}) starting in {current_zone}"
            )
        else:
            # Update existing record
            record = self.detection_records[object_id]

            # Update timestamps
            if timestamp:
                record["last_timestamp"] = timestamp

            # Update destination if zone changed
            if current_zone and current_zone != record["destination"]:
                old_destination = record["destination"]
                record["destination"] = current_zone
                record["zone_history"].append(current_zone)

                print(
                    f"📝 Updated ID {object_id} destination: {record['origin']} → {old_destination} → {current_zone}"
                )

            # Update moving status
            record["confirmed_moving"] = obj_info.get("confirmed_moving", False)

    def get_object_direction_info(self, object_id):
        """Get formatted origin and destination info for an object"""
        if object_id not in self.tracker.object_directions:
            return "Origin: Unknown", "Dest: Unknown"

        direction_info = self.tracker.object_directions[object_id]
        origin = direction_info["origin"] if direction_info["origin"] else "Unknown"
        destination = (
            direction_info["destination"]
            if direction_info["destination"]
            else "Tracking..."
        )

        return f"Origin: {origin}", f"Dest: {destination}"

    def draw_directional_boundaries(self, frame):
        """Draw X-shaped directional boundaries without filled zones"""
        height, width = frame.shape[:2]

        # Get current direction mapping
        current_directions = self.tracker.direction_manager.get_current_directions()

        # Define colors for direction labels
        direction_colors = {
            "North": (0, 255, 255),  # Yellow
            "South": (255, 0, 255),  # Magenta
            "East": (0, 255, 0),  # Green
            "West": (255, 165, 0),  # Orange
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
        cv2.line(
            frame,
            (0, border_thickness),
            (width, border_thickness),
            direction_colors[current_directions[0]],
            detection_line_thickness,
        )

        # Right detection border
        cv2.line(
            frame,
            (width - border_thickness, 0),
            (width - border_thickness, height),
            direction_colors[current_directions[1]],
            detection_line_thickness,
        )

        # Bottom detection border
        cv2.line(
            frame,
            (0, height - border_thickness),
            (width, height - border_thickness),
            direction_colors[current_directions[2]],
            detection_line_thickness,
        )

        # Left detection border
        cv2.line(
            frame,
            (border_thickness, 0),
            (border_thickness, height),
            direction_colors[current_directions[3]],
            detection_line_thickness,
        )

        # Add direction labels
        font_scale = 0.9
        font_thickness = 2

        # North label (top center)
        north_text = f"Top: {current_directions[0]}"
        (text_w, text_h), _ = cv2.getTextSize(
            north_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.putText(
            frame,
            north_text,
            (width // 2 - text_w // 2, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            direction_colors[current_directions[0]],
            font_thickness,
        )

        # South label (bottom center)
        south_text = f"Bottom: {current_directions[2]}"
        (text_w, text_h), _ = cv2.getTextSize(
            south_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.putText(
            frame,
            south_text,
            (width // 2 - text_w // 2, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            direction_colors[current_directions[2]],
            font_thickness,
        )

        # East label (right center)
        east_text = f"Right: {current_directions[1]}"
        cv2.putText(
            frame,
            east_text,
            (width - 180, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            direction_colors[current_directions[1]],
            font_thickness,
        )

        # West label (left center)
        west_text = f"Left: {current_directions[3]}"
        cv2.putText(
            frame,
            west_text,
            (10, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            direction_colors[current_directions[3]],
            font_thickness,
        )

    def preprocess_frames_batch(self, frames):
        processed_frames = []
        pil_images = []

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_images.append(pil_image)

        try:
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {
                k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
            }
            return inputs, pil_images
        except Exception as e:
            print(f"Batch preprocessing failed: {e}")
            inputs = self.processor(images=pil_images[0], return_tensors="pt")
            inputs = {
                k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
            }
            return inputs, [pil_images[0]]

    def detect_objects_batch(self, frames):
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

            batch_detections = []
            target_sizes = torch.tensor([img.size[::-1] for img in pil_images]).to(
                self.device
            )

            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )

            for result in results:
                detections = []
                for score, label, box in zip(
                    result["scores"], result["labels"], result["boxes"]
                ):
                    class_id = label.item()
                    confidence = score.item()

                    if (
                        class_id in self.target_classes
                        and confidence >= self.confidence_threshold
                    ):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        centroid_x = int((x1 + x2) / 2)
                        centroid_y = int((y1 + y2) / 2)
                        centroid = (centroid_x, centroid_y)
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        detections.append((centroid, class_id, confidence, bbox))

                batch_detections.append(detections)

            return batch_detections

        except Exception as e:
            print(
                f"Batch detection failed: {e}, falling back to single frame processing"
            )
            return [self.detect_objects_single(frame) for frame in frames]

    def detect_objects_single(self, frame):
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
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            class_id = label.item()
            confidence = score.item()

            if (
                class_id in self.target_classes
                and confidence >= self.confidence_threshold
            ):
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
        self.current_timestamp, self.current_timestamp_str = (
            self.timestamp_extractor.extract_timestamp_from_frame(frame)
        )

        # Detect objects
        detections = self.detect_objects_single(frame)

        # Update tracker
        tracked_objects = self.tracker.update(detections, self.current_timestamp)

        # Update detection records for all tracked objects (confirmed moving or candidates)
        for object_id, obj_info in tracked_objects.items():
            if object_id in self.tracker.object_directions:
                direction_info = self.tracker.object_directions[object_id]
                current_centroid = obj_info["centroid"]

                # Determine current zone
                current_zone = (
                    self.tracker.direction_manager.determine_direction_from_position(
                        current_centroid,
                        self.tracker.frame_width,
                        self.tracker.frame_height,
                    )
                )

                if current_zone:
                    # Update detection record (creates new or updates existing)
                    self.update_detection_record(
                        object_id, obj_info, current_zone, self.current_timestamp_str
                    )

        return tracked_objects

    def finalize_detection_record(self, object_id):
        """Finalize detection record when object is deregistered"""
        if object_id in self.detection_records:
            record = self.detection_records[object_id]

            # Only keep records for confirmed moving objects
            if not record.get("confirmed_moving", False):
                del self.detection_records[object_id]
                print(f"🗑️ Removed record for non-moving object ID {object_id}")
            else:
                # Mark as finalized
                record["status"] = "completed"

                # Calculate total journey info
                zone_changes = len(set(record["zone_history"])) - 1
                record["zone_changes"] = zone_changes
                record["final_path"] = f"{record['origin']} → {record['destination']}"

                print(
                    f"✅ Finalized record for ID {object_id}: {record['final_path']} ({zone_changes} zone changes)"
                )

    def draw_tracked_objects(self, frame, tracked_objects):
        """Draw enhanced annotations (only for moving objects)"""
        annotated_frame = frame.copy()

        # Draw directional boundaries first (so they appear behind objects)
        self.draw_directional_boundaries(annotated_frame)

        for object_id, obj_info in tracked_objects.items():
            if not obj_info.get("confirmed_moving", False):
                continue

            bbox = obj_info["bbox"]
            class_id = obj_info["class_id"]
            confidence = obj_info["confidence"]
            centroid = obj_info["centroid"]

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
                if timestamps["first_seen_timestamp"] and self.current_timestamp:
                    current_duration = (
                        self.current_timestamp - timestamps["first_seen_timestamp"]
                    ).total_seconds()
                    duration_text = f" ({current_duration:.1f}s)"

            # Get direction information
            origin_text, dest_text = self.get_object_direction_info(object_id)

            # Enhanced label with direction info
            label = f"ID:{object_id} {self.target_classes[class_id]} {confidence:.2f}{duration_text}"
            direction_label = f"{origin_text} to {dest_text}"

            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Calculate text dimensions for both labels
            (main_text_width, main_text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            (dir_text_width, dir_text_height), _ = cv2.getTextSize(
                direction_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale - 0.1,
                thickness - 1,
            )

            # Use wider text width for background
            max_text_width = max(main_text_width, dir_text_width)
            total_height = main_text_height + dir_text_height + 20

            # Draw label background (larger for two lines)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - total_height - 5),
                (x1 + max_text_width + 10, y1),
                color,
                -1,
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - total_height - 5),
                (x1 + max_text_width + 10, y1),
                (255, 255, 255),
                2,
            )

            # Draw main label (top line)
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, y1 - dir_text_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # Draw direction label (bottom line)
            cv2.putText(
                annotated_frame,
                direction_label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale - 0.1,
                (0, 255, 255),
                thickness - 1,
            )

            # Draw movement trail
            if object_id in self.tracker.object_history:
                points = self.tracker.object_history[object_id]
                if len(points) > 1:
                    for i in range(1, len(points)):
                        thickness = max(1, int(3 * (i / len(points))))
                        cv2.line(
                            annotated_frame, points[i - 1], points[i], color, thickness
                        )

        return annotated_frame

    def draw_timestamp_overlay(self, frame, frame_count):
        """Draw timestamp information overlay"""
        height, width = frame.shape[:2]

        # Timestamp panel
        panel_x = width - 400
        panel_y = height - 100
        cv2.rectangle(
            frame, (panel_x, panel_y), (width - 10, height - 10), (0, 0, 0), -1
        )
        cv2.rectangle(
            frame, (panel_x, panel_y), (width - 10, height - 10), (255, 255, 255), 2
        )

        # Display information
        timestamp_text = f"Extracted Timestamp: {self.current_timestamp_str if self.current_timestamp_str else 'None'}"
        cv2.putText(
            frame,
            timestamp_text,
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        frame_text = f"Frame: {frame_count}"
        cv2.putText(
            frame,
            frame_text,
            (panel_x + 10, panel_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if self.current_timestamp:
            video_time_text = (
                f"Video Time: {self.current_timestamp.strftime('%H:%M:%S')}"
            )
            cv2.putText(
                frame,
                video_time_text,
                (panel_x + 10, panel_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # GPU info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_text = f"GPU Memory: {gpu_memory:.2f}GB"
            cv2.putText(
                frame,
                gpu_text,
                (panel_x + 10, panel_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 165, 0),
                1,
            )

    def process_video(
        self,
        video_path,
        output_path=None,
        display=False,
        target_fps=30,
        save_preview_frames=True,
        preview_interval=25,
    ):
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

        print("=" * 80)
        print("GPU-OPTIMIZED DETR TRANSFORMER + MOTION DETECTION + DIRECTION TRACKING")
        print("=" * 80)
        print(f"Video Properties:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Original FPS: {original_fps:.2f}")
        print(f"  - Target FPS: {target_fps}")
        print(f"  - Frame Skip Ratio: {frame_skip}")
        print(f"  - Effective Processing FPS: {effective_fps:.2f}")
        print(f"  - Total Frames: {total_frames}")
        print(
            f"  - Direction Mapping: {self.tracker.direction_manager.get_current_mapping()}"
        )
        print("=" * 80)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

        frame_count = 0
        processed_frame_count = 0
        processing_start_time = time.time()
        detection_times = []

        # Create preview directory if needed
        if save_preview_frames:
            preview_dir = os.path.join(self.video_directory, "preview_frames")
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
                print(
                    f"PROCESSING FRAME {processed_frame_count} (actual frame {frame_count}/{total_frames})"
                )

                # Process frame
                tracked_objects = self.process_frame(frame)

                frame_processing_time = time.time() - frame_start_time
                detection_times.append(frame_processing_time)

                # Print results
                raw_detections = len(self.detect_objects_single(frame))
                moving_objects = len(
                    [
                        obj
                        for obj in tracked_objects.values()
                        if obj.get("confirmed_moving", False)
                    ]
                )
                candidates = len(tracked_objects) - moving_objects

                print(f"DETR raw detections: {raw_detections}")
                print(f"Moving objects: {moving_objects}")
                print(f"Candidates (checking motion): {candidates}")
                print(f"Processing time: {frame_processing_time:.3f}s")
                print(f"Extracted timestamp: {self.current_timestamp_str}")

                if self.current_timestamp:
                    print(
                        f"Parsed video timestamp: {self.current_timestamp.strftime('%d-%m-%Y %H:%M:%S')}"
                    )

                # Print GPU memory usage
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3
                    print(
                        f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached"
                    )

                # Print active objects
                for obj_id, obj_info in tracked_objects.items():
                    if obj_info.get("confirmed_moving", False):
                        if obj_id in self.tracker.object_timestamps:
                            timestamps = self.tracker.object_timestamps[obj_id]
                            if (
                                timestamps["first_seen_timestamp"]
                                and self.current_timestamp
                            ):
                                current_duration = (
                                    self.current_timestamp
                                    - timestamps["first_seen_timestamp"]
                                ).total_seconds()
                                class_name = self.target_classes[obj_info["class_id"]]
                                confidence = obj_info["confidence"]
                                print(
                                    f"  → ID {obj_id} ({class_name}): {current_duration:.1f}s | conf: {confidence:.3f} | MOVING"
                                )

                # Print candidates
                for obj_id, obj_info in tracked_objects.items():
                    if not obj_info.get("confirmed_moving", False):
                        class_name = self.target_classes[obj_info["class_id"]]
                        confidence = obj_info["confidence"]
                        print(
                            f"  ? ID {obj_id} ({class_name}): conf: {confidence:.3f} | CHECKING MOTION..."
                        )

                # Draw annotations
                annotated_frame = self.draw_tracked_objects(frame, tracked_objects)
                self.draw_timestamp_overlay(annotated_frame, processed_frame_count)

                # Handle display - TERMINAL COMPATIBLE
                if (
                    display
                    and save_preview_frames
                    and processed_frame_count % preview_interval == 0
                ):
                    preview_filename = os.path.join(
                        preview_dir, f"preview_frame_{processed_frame_count:06d}.jpg"
                    )
                    cv2.imwrite(preview_filename, annotated_frame)
                    print(f"📸 Saved preview: {preview_filename}")

                # Commented out cv2.imshow for terminal compatibility
                if display:
                    print("###")
                    if cv2.waitKey(1) & 0xFF == ord("q"):
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
                    processing_fps = (
                        processed_frame_count / elapsed if elapsed > 0 else 0
                    )
                    avg_detection_time = np.mean(detection_times)

                    print(f"\n{'*'*70}")
                    print(
                        f"PROGRESS: {progress:.1f}% | Processing FPS: {processing_fps:.1f}"
                    )
                    print(
                        f"Frames processed: {processed_frame_count}/{estimated_output_frames}"
                    )
                    print(f"Actual frames read: {frame_count}/{total_frames}")
                    print(f"Avg Detection Time: {avg_detection_time:.3f}s/frame")
                    print(
                        f"Moving Objects: {len([obj for obj in self.tracker.objects.values() if obj.get('confirmed_moving', False)])}"
                    )
                    print(
                        f"Candidates: {len([obj for obj in self.tracker.objects.values() if not obj.get('confirmed_moving', False)])}"
                    )
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

            # Clear GPU memory
            self.clear_gpu_memory()

            # Final statistics
            total_time = time.time() - processing_start_time
            time_saved = (
                (total_frames - processed_frame_count) * np.mean(detection_times)
                if detection_times
                else 0
            )

            print(f"\n{'='*80}")
            print("PROCESSING COMPLETED")
            print(f"{'='*80}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Frames processed: {processed_frame_count}/{total_frames}")
            print(f"Average detection time: {np.mean(detection_times):.3f}s/frame")
            print(f"Processing FPS: {processed_frame_count/total_time:.2f}")
            print(
                f"GPU acceleration: {'Used' if torch.cuda.is_available() else 'Not available'}"
            )
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
        print("\n" + "=" * 80)
        print("MOVING OBJECT DURATION ANALYSIS")
        print("=" * 80)

        if not self.tracker.object_durations:
            print("No completed moving object tracks found.")
            return

        class_durations = defaultdict(list)
        for obj_id, data in self.tracker.object_durations.items():
            class_name = data["class"]
            duration = data["duration"]
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
            print(
                f"  • Average Duration: {avg_duration:.2f} ± {std_duration:.2f} seconds"
            )
            print(f"  • Median Duration: {median_duration:.2f} seconds")
            print(f"  • Range: {min_duration:.2f}s - {max_duration:.2f}s")
            print(f"  • Total Time Visible: {total_class_duration:.2f} seconds")

        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  • Total Moving Objects: {total_objects}")
        print(f"  • Total Tracking Duration: {total_duration:.2f} seconds")
        print(
            f"  • Average Duration per Object: {total_duration/total_objects:.2f} seconds"
        )

        print(f"\nTRACKING TIMELINE:")
        print("-" * 60)
        for obj_id, data in sorted(self.tracker.object_durations.items()):
            first_seen = data["first_seen"].strftime("%H:%M:%S")
            last_seen = data["last_seen"].strftime("%H:%M:%S")
            duration = data["duration"]
            class_name = data["class"]
            print(
                f"ID {obj_id:2d} | {class_name:10s} | {duration:6.2f}s | {first_seen} → {last_seen}"
            )

        print("=" * 80)

    def export_detection_records(self):
        """Export detection records with dynamic origin→destination tracking"""
        if not self.detection_records:
            print("\n📊 No detection records to export.")
            return

        try:
            import pandas as pd

            # Convert dictionary to list for DataFrame
            records_list = []
            for obj_id, record in self.detection_records.items():
                # Only export confirmed moving objects
                if record.get("confirmed_moving", False):
                    export_record = {
                        "object_id": record["object_id"],
                        "object_type": record["object_type"],
                        "first_timestamp": record["first_timestamp"],
                        "last_timestamp": record["last_timestamp"],
                        "origin": record["origin"],
                        "destination": record["destination"],
                        # 'zone_changes': record.get('zone_changes', 0),
                        "zone_path": " to ".join(record["zone_history"]),
                        # 'status': record.get('status', 'active')
                    }
                    records_list.append(export_record)

            if not records_list:
                print("\n📊 No confirmed moving objects to export.")
                return

            # Create DataFrame
            df = pd.DataFrame(records_list)
            df = df.sort_values(["first_timestamp", "object_id"])

            # Generate filename
            timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.video_directory and os.path.exists(self.video_directory):
                csv_filename = os.path.join(
                    self.video_directory,
                    f"vehicle_tracking_dynamic_{timestamp_suffix}.csv",
                )
            else:
                csv_filename = f"vehicle_tracking_dynamic_{timestamp_suffix}.csv"

            # Export to CSV
            df.to_csv(csv_filename, index=False)
            print(f"\n📊 Detection records exported to: {csv_filename}")

            # Generate summary
            self.generate_tracking_summary(df, csv_filename)

        except ImportError:
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

    def generate_tracking_summary(self, df, csv_filename):
        """Generate comprehensive tracking summary"""
        summary_lines = []
        summary_lines.append("\n\nTRACKING SUMMARY:")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total moving objects tracked: {len(df)}")

        # Object type summary
        type_counts = df["object_type"].value_counts()
        summary_lines.append("\nObject type distribution:")
        for obj_type, count in type_counts.items():
            summary_lines.append(f"  {obj_type}: {count}")

        # Origin-Destination flow analysis
        if "origin" in df.columns and "destination" in df.columns:
            summary_lines.append("\nOrigin to Destination flows:")

            # Group by origin-destination pairs
            flow_counts = (
                df.groupby(["origin", "destination"]).size().reset_index(name="count")
            )
            flow_counts = flow_counts.sort_values("count", ascending=False)

            for _, row in flow_counts.iterrows():
                origin = row["origin"]
                destination = row["destination"]
                count = row["count"]

                if origin == destination:
                    summary_lines.append(f"  {origin} (stayed): {count}")
                else:
                    summary_lines.append(f"  {origin} to {destination}: {count}")

        # Time range analysis
        if "first_timestamp" in df.columns and "last_timestamp" in df.columns:
            summary_lines.append(f"\nTime range:")
            summary_lines.append(f" Video start: {df['first_timestamp'].min()}")
            summary_lines.append(f" Video stop: {df['last_timestamp'].max()}")

        # Append summary to CSV file
        with open(csv_filename, "a", encoding="utf-8") as f:
            for line in summary_lines:
                f.write(line + "\n")

        # Print summary to console
        print("\n".join(summary_lines))

    def export_detection_records_manual(self):
        """Manual CSV export without pandas"""
        if not self.detection_records:
            return

        try:
            timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = os.path.join(
                self.video_directory, f"vehicle_tracking_dynamic_{timestamp_suffix}.csv"
            )

            # Filter for confirmed moving objects only
            moving_records = {
                k: v
                for k, v in self.detection_records.items()
                if v.get("confirmed_moving", False)
            }

            if not moving_records:
                print("\n📊 No confirmed moving objects to export.")
                return

            with open(csv_filename, "w", newline="") as csvfile:
                headers = [
                    "object_id",
                    "object_type",
                    "first_timestamp",
                    "last_timestamp",
                    "origin",
                    "destination",
                    "zone_path",
                ]
                csvfile.write(",".join(headers) + "\n")

                for obj_id, record in moving_records.items():
                    zone_path = " → ".join(record["zone_history"])

                    row = [
                        str(record["object_id"]),
                        record["object_type"],
                        record["first_timestamp"],
                        record["last_timestamp"],
                        record["origin"],
                        record["destination"],
                        zone_path,
                    ]
                    csvfile.write(",".join(row) + "\n")

            print(f"\n📊 Detection records exported to: {csv_filename}")
            print(f"  • Total records: {len(moving_records)}")

            # Generate summary directly without class
            self.generate_tracking_summary_manual(moving_records, csv_filename)

        except Exception as e:
            print(f"\n⚠ Manual CSV export failed: {e}")

    def generate_tracking_summary_manual(self, moving_records, csv_filename):
        """Generate comprehensive tracking summary for manual export"""
        summary_lines = []
        summary_lines.append("\n\nTRACKING SUMMARY:")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total moving objects tracked: {len(moving_records)}")

        # Object type summary
        type_counts = {}
        for record in moving_records.values():
            obj_type = record["object_type"]
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        summary_lines.append("\nObject type distribution:")
        for obj_type, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            summary_lines.append(f"  {obj_type}: {count}")

        # Origin-Destination flow analysis
        summary_lines.append("\nOrigin to Destination flows:")

        # Group by origin-destination pairs manually
        flow_counts = {}
        for record in moving_records.values():
            origin = record["origin"]
            destination = record["destination"]
            key = (origin, destination)
            flow_counts[key] = flow_counts.get(key, 0) + 1

        # Sort by count (descending)
        sorted_flows = sorted(flow_counts.items(), key=lambda x: x[1], reverse=True)

        for (origin, destination), count in sorted_flows:
            if origin == destination:
                summary_lines.append(f"  {origin} (stayed): {count}")
            else:
                summary_lines.append(f"  {origin} to {destination}: {count}")

        # Time range analysis
        first_timestamps = [
            record["first_timestamp"] for record in moving_records.values()
        ]
        last_timestamps = [
            record["last_timestamp"] for record in moving_records.values()
        ]

        summary_lines.append(f"\nTime range:")
        summary_lines.append(f" Video start: {min(first_timestamps)}")
        summary_lines.append(f" Video stop: {max(last_timestamps)}")

        # Append summary to CSV file
        with open(csv_filename, "a", encoding="utf-8") as f:
            for line in summary_lines:
                f.write(line + "\n")

        # Print summary to console
        print("\n".join(summary_lines))

    def export_duration_data(self):
        """Export duration data to CSV"""
        if not self.tracker.object_durations:
            print("\n📊 No duration data to export.")
            return

        try:
            import pandas as pd

            export_data = []
            for obj_id, data in self.tracker.object_durations.items():
                export_data.append(
                    {
                        "object_id": obj_id,
                        "class": data["class"],
                        "duration_seconds": data["duration"],
                        "first_seen": data["first_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                        "last_seen": data["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                        "movement_confirmed": True,
                    }
                )

            if export_data:
                df = pd.DataFrame(export_data)
                df = df.sort_values(
                    ["class", "duration_seconds"], ascending=[True, False]
                )

                timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = os.path.join(
                    self.video_directory,
                    f"moving_vehicle_durations_{timestamp_suffix}.csv",
                )

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
            timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = os.path.join(
                self.video_directory, f"moving_vehicle_durations_{timestamp_suffix}.csv"
            )

            with open(csv_filename, "w", newline="") as csvfile:
                headers = [
                    "object_id",
                    "class",
                    "duration_seconds",
                    "first_seen",
                    "last_seen",
                    "movement_confirmed",
                ]
                csvfile.write(",".join(headers) + "\n")

                for obj_id, data in self.tracker.object_durations.items():
                    row = [
                        str(obj_id),
                        data["class"],
                        str(data["duration"]),
                        data["first_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                        data["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                        "True",
                    ]
                    csvfile.write(",".join(row) + "\n")

            print(f"\n📊 Duration data exported to: {csv_filename}")

        except Exception as e:
            print(f"\n⚠ Manual duration CSV export failed: {e}")

    # The high-level frame/video processing methods remain in detr_motion.py to avoid duplication.


__all__ = ["GPUOptimizedDETRDetector"]
