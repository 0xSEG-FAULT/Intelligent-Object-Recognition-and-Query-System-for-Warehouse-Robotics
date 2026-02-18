import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
import numpy as np

class WarehouseClassifier:
    def __init__(self, model_path="models/warehouse_resnet_final"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Maps model output ID to your specific warehouse categories [cite: 34]
        self.id2label = {0: 'Fragile', 1: 'Heavy', 2: 'Hazardous'}
        
        try:
            # Load fine-tuned model if available
            self.model = AutoModelForImageClassification.from_pretrained(model_path)
            self.processor = AutoImageProcessor.from_pretrained(model_path)
        except OSError:
            print("[WARNING] Trained model not found. Loading base ResNet-50 (untrained on your classes).")
            # Fallback for testing before training is complete
            self.model = AutoModelForImageClassification.from_pretrained(
                "microsoft/resnet-50", 
                num_labels=3, 
                ignore_mismatched_sizes=True
            )
            self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_numpy):
        """
        Classifies an OpenCV image (numpy array) into Fragile/Heavy/Hazardous.
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        if image_numpy.size == 0: return "Unknown"
        
        img = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        
        # Inference
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()
            
        return self.id2label.get(predicted_idx, "Unknown")