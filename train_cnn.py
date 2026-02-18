import os
import shutil
import random
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer
)
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
# fiftyone is imported conditionally inside prepare_data() to avoid version conflicts
# (mongoengine/pymongo incompatibility) if data already exists

# MAPPING CONFIGURATION
# We map specific COCO classes to your 3 target categories
CLASS_MAPPING = {
    "Fragile": ["bottle", "wine glass", "cup", "vase"],
    "Heavy": ["suitcase", "couch", "refrigerator", "bed"],
    "Hazardous": ["laptop", "tv", "knife", "scissors"]
}

def prepare_data():
    """Downloads and organizes COCO data into class folders."""
    base_dir = Path("data/mapped_coco")
    if base_dir.exists():
        print("Data directory exists. Skipping download.")
        return

    # Import fiftyone only if needed (to avoid version conflicts with mongoengine/pymongo)
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as e:
        print(f"Error: FiftyOne is required to download COCO data but failed to import: {e}")
        print("Data directory not found, and FiftyOne could not be loaded.")
        print("Please ensure your data is in data/mapped_coco/train and data/mapped_coco/test")
        raise

    print("Downloading COCO subset via FiftyOne...")
    all_classes = [c for sublist in CLASS_MAPPING.values() for c in sublist]
    
    # Download Validation set (smaller/faster for assignment)
    dataset = foz.load_zoo_dataset(
        "coco-2017",
         split="validation", # Use validation for a smaller subset, or train for the full set
        label_types=["detections"],
        classes=all_classes,
        max_samples=600           # Uncomment to limit samples for faster testing during development
    )

    # Organize into folders
    for sample in dataset:
        # Find the largest object in the image
        best_label = None
        max_area = 0
        for det in sample.ground_truth.detections:
            w, h = det.bounding_box[2], det.bounding_box[3]
            if w * h > max_area and det.label in all_classes:
                max_area = w * h
                best_label = det.label
        
        if best_label:
            # Map COCO label to Target Label
            target_class = next(k for k, v in CLASS_MAPPING.items() if best_label in v)
            
            # Save
            split = "train" if random.random() < 0.8 else "test"
            dest = base_dir / split / target_class
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(sample.filepath, dest / Path(sample.filepath).name)

def train():
    prepare_data()
    
    # Load from folders
    dataset = load_dataset("imagefolder", data_dir="data/mapped_coco")
    
    # Preprocessing
    checkpoint = "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    normalize = Compose([
        Resize((224, 224)),
        ToTensor(), 
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    def transforms(examples):
        examples["pixel_values"] = [normalize(img.convert("RGB")) for img in examples["image"]]
        return examples

    processed_dataset = dataset.map(transforms, batched=True, remove_columns=["image"])
    
    # Model Setup
    labels = dataset["train"].features["label"].names
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
        ignore_mismatched_sizes=True
    )

    # Training Args
    args = TrainingArguments(
        output_dir="models/checkpoints",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=processor,
    )

    trainer.train()
    trainer.save_model("models/warehouse_resnet_final")
    print("Training Complete. Model saved to models/warehouse_resnet_final")

if __name__ == "__main__":
    train()