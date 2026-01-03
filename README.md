# 🔬 Medical AI: Cell Segmentation and Tracking with U-Net

This project focuses on the automated segmentation and tracking of cells in phase-contrast microscopy images (specifically the `PhC-C2DH-U373` cell line) using a **U-Net** deep learning architecture and a **Centroid Tracking** algorithm.

---

## 🚀 Key Features
* **Deep Learning Segmentation:** High-precision cell detection using a U-Net model implemented in PyTorch.
* **Real-time Tracking:** Assigns a unique, persistent ID to each cell and tracks them across consecutive frames.
* **Trajectory Visualization:** Draws historical migration paths to visualize cell movement over time.
* **Velocity Estimation:** Calculates real-time speed of cells (pixels per frame).
* **Automated Video Output:** Encodes and saves the final analysis as an `.mp4` video file using OpenCV.

---

## 📊 Dataset
The project utilizes the `PhC-C2DH-U373` dataset from the **Cell Tracking Challenge**.
* **Type:** Phase-contrast microscopy.
* **Content:** 115 frames of glioblastoma-astrocytoma cells.
* **Ground Truth:** Manually annotated segmentation masks.

---

## 🧠 Model & Training
The core of the project is a **U-Net** convolutional neural network.
* **Loss Function:** Combined Binary Cross Entropy (BCE) and Dice Loss for robust small-object segmentation.
* **Hyperparameters:**
  * Epochs: 30 - 50
  * Learning Rate: 1e-3
  * Optimizer: Adam
* **Post-Processing:** Morphological "Closing" operations to merge cell fragments and remove noise.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your_username/U373-Cell-Tracking.git](https://github.com/your_username/U373-Cell-Tracking.git)
   cd U373-Cell-Tracking
