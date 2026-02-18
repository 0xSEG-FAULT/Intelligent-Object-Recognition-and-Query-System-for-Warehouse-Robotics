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