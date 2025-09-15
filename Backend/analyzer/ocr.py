"""Timestamp extraction utilities (moved from detr_motion.py).

This module contains the TimestampExtractor class, refactored out of the
monolithic detr_motion.py without changing functionality.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple

import cv2
import pytesseract


class TimestampExtractor:
    """Extract and parse timestamps from video frames - Enhanced to extract both date and time"""

    def __init__(self, roi_height_percent: float = 0.1, roi_width_percent: float = 0.4):
        """
        Initialize timestamp extractor
        Args:
                roi_height_percent: Height of ROI as percentage of frame height (default: 0.1 = top 10%)
                roi_width_percent: Width of ROI as percentage of frame width (default: 0.4 = left 40%)
        """
        self.roi_height_percent = roi_height_percent
        self.roi_width_percent = roi_width_percent
        self.last_known_timestamp: Optional[datetime] = None
        self.timestamp_format = "%d-%m-%Y %H:%M:%S"

    def extract_timestamp_from_frame(self, frame) -> Tuple[Optional[datetime], str]:
        """
        Extract timestamp from the top-left corner of a frame using OCR.
        Returns both datetime object and string representation (or raw text on failure).
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
            _, roi_thresh = cv2.threshold(
                roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Increase contrast
            roi_thresh = cv2.bitwise_not(roi_thresh)

            # Use OCR to extract text - Enhanced for better date/time recognition
            custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-: /"
            text = pytesseract.image_to_string(roi_thresh, config=custom_config)

            # Clean up the text
            text = text.strip().replace("\n", " ")

            # Enhanced patterns to capture both date and time
            patterns_and_formats = [
                (
                    r"(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})",
                    "%d-%m-%Y %H:%M:%S",
                ),  # DD-MM-YYYY HH:MM:SS
                (
                    r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})",
                    "%d/%m/%Y %H:%M:%S",
                ),  # DD/MM/YYYY HH:MM:SS
                (
                    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
                    "%Y-%m-%d %H:%M:%S",
                ),  # YYYY-MM-DD HH:MM:SS
                (
                    r"(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2})",
                    "%d-%m-%Y %H:%M",
                ),  # DD-MM-YYYY HH:MM
                (
                    r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})",
                    "%d/%m/%Y %H:%M",
                ),  # DD/MM/YYYY HH:MM
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
            time_pattern = r"(\d{2}:\d{2}:\d{2})"
            time_match = re.search(time_pattern, text)
            if time_match and self.last_known_timestamp:
                time_str = time_match.group(1)
                # Combine with last known date
                date_str = self.last_known_timestamp.strftime("%d-%m-%Y")
                full_timestamp_str = f"{date_str} {time_str}"
                try:
                    timestamp = datetime.strptime(
                        full_timestamp_str, "%d-%m-%Y %H:%M:%S"
                    )
                    self.last_known_timestamp = timestamp
                    return timestamp, full_timestamp_str
                except ValueError:
                    pass

            # If no pattern matches, return raw text for debugging
            raw_text = f"{text}" if text else "No timestamp detected"
            return None, raw_text

        except Exception as e:
            return None, f"OCR Error: {str(e)}"

    def extract_timestamp(self, frame) -> Optional[datetime]:
        """Extract timestamp - wrapper for compatibility with existing code"""
        timestamp_obj, _ = self.extract_timestamp_from_frame(frame)
        return timestamp_obj


__all__ = ["TimestampExtractor"]
