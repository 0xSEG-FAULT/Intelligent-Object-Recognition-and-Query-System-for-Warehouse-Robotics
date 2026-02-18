import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        # Parameters for edge detection
        self.min_area = 1500  # Minimum area to be considered an object

    def detect_objects(self, frame):
        """
        Detects objects using Canny Edge Detection and Contour Analysis.
        Returns: 
        - detected_objects: List of dicts with bbox, center, and ROI
        - processed_frame: Frame with visualizations (edges)
        """
        # 1. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Edge Detection [cite: 17]
        edged = cv2.Canny(blurred, 50, 150)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            # 4. Bounding Box & Center [cite: 22, 24]
            x, y, w, h = cv2.boundingRect(c)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Extract ROI (Region of Interest) for ML
            roi = frame[y:y+h, x:x+w]
            
            detected_objects.append({
                "bbox": (x, y, w, h),
                "center": (center_x, center_y),
                "dims": (w, h), # Approximate dimensions in pixels [cite: 23]
                "roi": roi
            })

        return detected_objects, edged

    def segment_by_color(self, roi):
        """
        Performs color-based segmentation on a Region of Interest (ROI).
        Returns the dominant color name and a mask showing the color distribution.
        
        Uses HSV color space for robust color detection across different lighting conditions.
        """
        if roi.size == 0:
            return "Unknown", None
        
        # Convert ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV (Hue, Saturation, Value)
        # Hue range: 0-180 in OpenCV (scaled down from 0-360)
        color_ranges = {
            "Red": [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                    (np.array([170, 50, 50]), np.array([180, 255, 255]))],  # Red wraps around
            "Orange": [(np.array([10, 50, 50]), np.array([25, 255, 255]))],
            "Yellow": [(np.array([25, 50, 50]), np.array([35, 255, 255]))],
            "Green": [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
            "Cyan": [(np.array([85, 50, 50]), np.array([100, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Purple": [(np.array([130, 50, 50]), np.array([170, 255, 255]))],
        }
        
        # Find the dominant color
        max_pixels = 0
        dominant_color = "Unknown"
        color_masks = {}
        
        for color_name, ranges in color_ranges.items():
            mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
            
            # For red, combine both ranges
            if color_name == "Red" and len(ranges) > 1:
                mask_2 = cv2.inRange(hsv, ranges[1][0], ranges[1][1])
                mask = cv2.bitwise_or(mask, mask_2)
            
            # Count pixels in this color range
            pixel_count = cv2.countNonZero(mask)
            color_masks[color_name] = mask
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color_name
        
        # Return dominant color and its mask
        mask = color_masks.get(dominant_color, None)
        return dominant_color, mask