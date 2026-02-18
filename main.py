import cv2
import torch
from src.cv_module import ObjectDetector
from src.ml_module import WarehouseClassifier
from src.rag_module import RoboticsRAG
from src.tracking_module import CentroidTracker

def main():
    print("Loading Modules...")
    # Initialize components
    detector = ObjectDetector()
    classifier = WarehouseClassifier()
    rag = RoboticsRAG()
    ct = CentroidTracker(maxDisappeared=40)  # Object tracking with ID assignment
    
    cap = cv2.VideoCapture(0) # Use 0 for Webcam, or path to video file
    
    current_object_label = None

    print("\n--- Warehouse Robot System Ready ---")
    print("Press 'q' to quit.")
    print("Press 'space' to ask a question about the detected object.")
    print("Press 't' for COLOR TEST MODE (tests all 3 categories).")
    print("Tracking: Objects are automatically tracked across frames with unique IDs.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect [cite: 73]
        objects, edges = detector.detect_objects(frame)
        
        # Extract centroids from detected objects for tracking
        centroids = []
        if objects:
            centroids = [obj['center'] for obj in objects]
        
        # 2. Update tracker with new centroids
        tracked_objects = ct.update(centroids)
        
        # Reset label if no objects found
        if not objects:
            current_object_label = None
        
        # 3. Process each detected object
        for obj in objects:
            x, y, w, h = obj['bbox']
            center_x, center_y = obj['center']
            roi = obj['roi']
            
            # Classify (Only if object is large enough) [cite: 74]
            if w > 50 and h > 50:
                # Crop and Predict
                label = classifier.predict(roi)
                # Clean the label: strip whitespace, handle encoding
                label = str(label).strip()
                current_object_label = label
                
                # Debug: Print label for troubleshooting
                print(f"[DEBUG] Raw label: '{label}' | Type: {type(label)}")
                
                # Define color for each category (B, G, R format)
                color_map = {
                    "Fragile": (0, 255, 255),      # Yellow
                    "Heavy": (255, 0, 0),          # Blue
                    "Hazardous": (0, 0, 255),      # Red
                }
                
                # Get color based on label, default to gray if unknown
                color = color_map.get(label, (128, 128, 128))
                
                print(f"[DEBUG] Label matched: {label} → Color: {color}")
                
                # Draw semi-transparent filled rectangle for better visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)  # -1 fills the rectangle
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # Draw thick bounding box outline with category color
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
                
                # Find the object ID for this centroid (match by position)
                objectID = None
                for oid, (ox, oy) in tracked_objects.items():
                    if abs(ox - center_x) < 5 and abs(oy - center_y) < 5:
                        objectID = oid
                        break
                
                # Compute safe text positions to avoid overlap
                id_y = max(15, y - 60)
                label_y = max(15, y - 40)
                dims_y = max(15, y - 20)
                center_text_y = min(frame.shape[0] - 10, y + h + 25)

                # Display Object ID (above the label)
                if objectID is not None:
                    cv2.putText(frame, f"ID: {objectID}", (x, id_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Display Classification Label (main category)
                cv2.putText(frame, f"{label}", (x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                # Display Dimensions (Width x Height in pixels)
                cv2.putText(frame, f"Dims: {w}x{h}px", (x, dims_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Display Center Coordinates
                cv2.putText(frame, f"Center: ({center_x}, {center_y})", (x, center_text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw a small circle at the center point with thicker border
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                cv2.circle(frame, (center_x, center_y), 7, (255, 255, 255), 1)
                
                # Draw tracking trail (movement history)
                if objectID is not None:
                    track_history = ct.get_track_history(objectID)
                    if len(track_history) > 1:
                        # Draw lines connecting previous positions
                        for i in range(1, len(track_history)):
                            p1 = tuple(map(int, track_history[i-1]))
                            p2 = tuple(map(int, track_history[i]))
                            # Fade color intensity based on age (older = lighter)
                            alpha = int(200 * (i / len(track_history)))
                            fade_color = tuple(int(c * alpha / 255) for c in color)
                            cv2.line(frame, p1, p2, fade_color, 2)

        # Show Output
        cv2.imshow("Robot Vision", frame)
        # cv2.imshow("Edges", edges) # Optional debug view

        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            # Test mode: display all color mappings
            print("\n[TEST MODE] Color Mappings:")
            print("  FRAGILE  (Yellow) → (0, 255, 255)")
            print("  HEAVY    (Blue)   → (255, 0, 0)")
            print("  HAZARDOUS (Red)  → (0, 0, 255)")
            print("Move different colored objects in front of camera to test!")
            print("Press any key to continue...\n")
            cv2.waitKey(0)
        elif key == ord(' '):
            # 3. RAG Query Trigger [cite: 75]
            if current_object_label:
                print(f"\n[System] Focus: {current_object_label}")
                query = input("[User] Enter your question: ")
                response = rag.query(query, current_object_label)
                answer = response.split("<|assistant|>")[-1].strip()
                print(f"[RAG Response] {answer}\n")
            else:
                print("[System] No object detected to ask about.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()