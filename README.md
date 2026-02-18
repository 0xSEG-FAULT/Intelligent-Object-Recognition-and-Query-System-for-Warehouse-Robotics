#  Intelligent Object Recognition & Query System

An end-to-end warehouse robotics prototype integrating:

-   Computer Vision (OpenCV-based detection & tracking)
-   Deep Learning (ResNet-50 Transfer Learning)
-   Retrieval-Augmented Generation (RAG) with TinyLlama)

The system detects objects, classifies them into actionable categories
(Fragile, Heavy, Hazardous), and generates grounded safety instructions
using a local LLM pipeline optimized for edge deployment.

------------------------------------------------------------------------

# Setup & Installation

## 1️⃣ Prerequisites

-   Python 3.10+
-   Git installed
-   (Optional) NVIDIA GPU with CUDA
-   \~3GB free disk space

------------------------------------------------------------------------

## 2️⃣ Clone the Repository

``` bash
git clone https://github.com/0xSEG-FAULT/Intelligent-Object-Recognition-and-Query-System-for-Warehouse-Robotics.git
cd Intelligent-Object-Recognition
```

------------------------------------------------------------------------

## 3️⃣ Create Virtual Environment

### Windows (PowerShell)


python -m venv venv
.venv\Scripts\activate


### macOS / Linux


python3 -m venv venv
source venv/bin/activate


------------------------------------------------------------------------

## 4️⃣ Install Dependencies(in terminal)

### CPU Installation


pip install -r requirements.txt


### GPU Installation (Recommended)


pip install -r requirements.txt
pip install -r requirements-gpu.txt


------------------------------------------------------------------------

# 🚀 How to Run

## 1️⃣ Train the Model


python train_cnn.py


-   Downloads COCO subset via FiftyOne
-   Fine-tunes ResNet-50 on 3 classes
-   Saves model to models/warehouse_resnet_final

------------------------------------------------------------------------

## 2️⃣ Start the Robot


python main.py


Controls:

  Key        Function
  ---------- ------------------------------------
  Spacebar   Ask question about detected object
  t          Toggle test mode
  q          Quit

------------------------------------------------------------------------

## 3️⃣ Evaluate Performance


python evaluate_model.py


Outputs: - results/performance_report.txt - results/confusion_matrix.png

------------------------------------------------------------------------

# Challenges Faced & Solutions
1. Infrastructure & Dependencies
Dependency Conflicts: Installing the full ML stack (Torch, Transformers, LangChain) initially caused pip resolution errors due to conflicting version requirements.

Solution: I adopted a staged installation approach, installing PyTorch first to lock the CUDA version, followed by the Hugging Face stack, and finally the RAG components. I also created separate requirements.txt (CPU) and requirements-gpu.txt files to ensure cross-platform compatibility.

OpenCV GUI Crashes: The system initially crashed with The function is not implemented errors because standard ML libraries auto-installed opencv-python-headless.

Solution: I explicitly uninstalled the headless version and forced the installation of the standard opencv-python package to enable real-time camera visualization (cv2.imshow).

2. Model Training & Data Pipeline
Class Imbalance & Mode Collapse: Due to hardware constraints (laptop GPU) and the 4-day timeline, I trained on a small subset of COCO (~5,000 images). This led to Mode Collapse on the "Hazardous" class, where the model favored the majority classes ("Fragile" and "Heavy") to minimize loss.

Solution: I prioritized a functional end-to-end pipeline over perfect accuracy. I documented this behavior in the performance report, noting that a full-scale training run with Class Weighting (e.g., WeightedRandomSampler) would resolve this in a production environment.

Data Processing Errors: The datasets library caused KeyError: 'image' crashes during batch processing because raw image data was being dropped prematurely.

Solution: I refactored the preprocessing pipeline to apply transforms using .map() and explicitly removed the raw columns after tensor conversion was complete.

3. RAG System & LLM Integration
Context Hallucination: Early versions of the RAG system would provide "Hazardous" handling advice for "Fragile" items because it retrieved the entire protocol file as context.

Solution: I implemented Strict Context Prompting in the LLM. The system now dynamically inserts the detected object label into the prompt (e.g., "The user is asking about a Fragile item...") and instructs the model to ignore unrelated protocols.

Generation Parameter Conflicts: The LLM pipeline threw warnings about temperature being used without sampling enabled.

Solution: I optimized the generation config to use deterministic outputs (do_sample=False) for consistent grading results, while keeping the architecture flexible for creative sampling if needed.

4. Version Control & Deployment
Repository Size Limits: Attempting to push the trained model (~100MB+) blocked the Git push due to file size limits.

Solution: I implemented a strict .gitignore policy for models/ and data/ directories and used git rm --cached to scrub large files from the history, ensuring a lightweight and professional repository structure.

5. Computer Vision StabilityLighting Sensitivity & Noise: The Canny Edge Detection algorithm was initially too sensitive, detecting shadows and floor patterns as objects, creating "ghost" detections.Solution: I implemented a multi-stage preprocessing pipeline. First, Gaussian Blur ($5 \times 5$ kernel) is applied to smooth out high-frequency noise. Second, I added a strictly tuned Area Threshold (ignoring contours $< 1,500$ pixels), which effectively filters out shadows and small debris, ensuring only physical objects are tracked.
6. Object Tracking & Identity PersistenceID Switching (The "Flickering" Problem): When objects moved quickly or momentarily left the frame, the Centroid Tracker would lose them and assign a new ID (e.g., ID 1 becomes ID 2), making tracking history unreliable.Solution: I tuned the maxDisappeared parameter in the tracking module. By increasing the persistence window to 40 frames, the system now "remembers" an object's location during brief occlusions or detection failures, maintaining a stable ID even if the object is momentarily undetected.
7. Real-Time Inference Latency (FPS Drop)Performance Bottleneck: Running the heavy ResNet-50 classifier on every single video frame caused the system to lag (drop below 10 FPS), making the video feed choppy.Solution: I decoupled detection from classification. The lightweight Object Detector runs on every frame to maintain smooth tracking, while the heavy ResNet Classifier is triggered only when a valid Region of Interest (ROI) is confirmed. This "lazy evaluation" strategy kept the interface responsive while still updating predictions in real-time.
8. GPU Memory Management (OOM Errors)Resource Contention: Loading both the Vision Model (ResNet-50) and the Language Model (TinyLlama-1.1B) simultaneously on a single laptop GPU (4GB VRAM) initially caused CUDA Out Of Memory errors.Solution: I optimized model precision. I loaded the LLM in torch.float16 (half-precision) mode and ensured that the RAG pipeline releases unnecessary tensors after generation. This reduced VRAM usage by ~40%, allowing both models to coexist in the same GPU memory space.
9. RAG Retrieval PrecisionGeneric Context Retrieval: Initially, queries like "How to handle this?" retrieved mixed safety protocols because the vector similarity scores for "safety" were too uniform across different documents.Solution: I improved the Knowledge Base structure. Instead of one large text block, I separated protocols into distinct files (fragile.txt, heavy.txt) with clear, unique headers. I also increased the chunk_overlap to 50 tokens, ensuring that splitting the text didn't cut off crucial context keywords, leading to higher confidence during vector retrieval.