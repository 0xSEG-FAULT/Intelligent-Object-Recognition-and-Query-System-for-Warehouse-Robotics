import cv2
import torch
from src.cv_module import ObjectDetector
from src.ml_module import WarehouseClassifier
from src.rag_module import RoboticsRAG

def main():
    print("Loading Modules...")
    # Initialize components
    detector = ObjectDetector()
    classifier = WarehouseClassifier()
    rag = RoboticsRAG()
    
    cap = cv2.VideoCapture(0) # Use 0 for Webcam, or path to video file
    
    current_object_label = None

    print("\n--- Warehouse Robot System Ready ---")
    print("Press 'q' to quit.")
    print("Press 'space' to ask a question about the detected object.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect [cite: 73]
        objects, edges = detector.detect_objects(frame)
        
        # Reset label if no objects found
        if not objects:
            current_object_label = None
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            center_x, center_y = obj['center']
            
            # 2. Classify (Only if object is large enough) [cite: 74]
            if w > 50 and h > 50:
                # Crop and Predict
                label = classifier.predict(obj['roi'])
                current_object_label = label
                
                # Draw Box & Label
                color = (0, 255, 0)
                if label == "Hazardous": color = (0, 0, 255) # Red for danger
                elif label == "Fragile": color = (0, 255, 255) # Yellow
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display Classification Label
                cv2.putText(frame, f"{label}", (x, y-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Display Dimensions (Width x Height in pixels)
                cv2.putText(frame, f"Dims: {w}x{h}px", (x, y-25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                
                # Display Center Coordinates
                cv2.putText(frame, f"Center: ({center_x}, {center_y})", (x, y+h+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                
                # Draw a small circle at the center point
                cv2.circle(frame, (center_x, center_y), 5, color, -1)

        # Show Output
        cv2.imshow("Robot Vision", frame)
        # cv2.imshow("Edges", edges) # Optional debug view

        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
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