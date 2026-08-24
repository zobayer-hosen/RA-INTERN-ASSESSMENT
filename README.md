# Brain Tumor Classification — Complete Experiments (Fixed v3)

This project compares a centralized machine learning baseline with federated learning (FL) paradigms for classifying brain tumors from MRI scans. It evaluates performance across three experimental configurations using a ResNet50 model:
1. **Experiment 1 (Centralized):** Training on the entire centralized dataset (15 epochs).
2. **Experiment 2 (Federated IID):** Training across 3 clients with identically and independently distributed data (5 rounds × 3 local epochs = 15 total epochs).
3. **Experiment 3 (Federated Non-IID):** Training across 3 clients with heterogeneous, class-imbalanced data splits (5 rounds × 3 local epochs = 15 total epochs).

---

## Research & Problem Statement
Brain tumor classification from MRI scans is essential for automated medical diagnosis and patient treatment paths. However, centralizing medical data across different clinics poses severe privacy, security, and regulatory challenges (e.g., HIPAA). 

Federated Learning (FL) enables collaborative model training without aggregating sensitive clinical data. Each client trains a model locally on its own hardware, and only model parameters are shared with a central server for aggregation. This study evaluates:
- The performance gap between centralized baselines and federated setups.
- The impact of data heterogeneity (IID vs. Non-IID partitions) on global model convergence and classification accuracy.
- The communication bandwidth overhead associated with local parameter updates.

---

## Dataset Information
The experiments are conducted on the **Nickparvar dataset** consisting of brain MRI images categorized into 4 classes:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

### Dataset Structure
- **Total Images:** 7,200
  - **Training Set (5,600 images):** 1,400 images per class (balanced)
  - **Testing Set (1,600 images):** 400 images per class (balanced)

---

## Preprocessing
Images are preprocessed using `torchvision.transforms` to remove watermarks/borders and introduce robustness during training:

### Training Transforms
1. **Resize:** Rescale images to $256 \times 256$ pixels.
2. **Center Crop:** Crop to $224 \times 224$ pixels to eliminate border artifacts.
3. **Random Horizontal Flip:** Apply horizontal flip for data augmentation.
4. **Random Rotation:** Rotate randomly by up to $15^\circ$.
5. **Color Jitter:** Adjust brightness and contrast by a factor of 0.2.
6. **Normalize:** Scale image pixels to match ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).

### Testing Transforms
1. **Resize:** Rescale to $256 \times 256$ pixels.
2. **Center Crop:** Crop to $224 \times 224$ pixels.
3. **Normalize:** Apply ImageNet statistics.

---

## Model Architecture
- **Backbone:** ResNet50 pre-trained on ImageNet-1K (`weights=models.ResNet50_Weights.IMAGENET1K_V1`).
- **Feature Freezing:** Feature extractor layers are frozen except for `layer4` and the classification head, allowing targeted fine-tuning on medical scans.
- **Classification Head:** A custom sequential head is appended:
  - **Dropout Layer:** $p = 0.5$ (preventing overfitting).
  - **Linear Layer:** Maps the 2,048 feature dimensions to the 4 output classes.
- **Trainable Parameters:** ~14,972,866 parameters (representing `layer4` and the `fc` classifier head).

---

## Training Methodology

### Configuration
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Optimizer:** Adam (applied to trainable parameters)
- **Loss Function:** Cross-Entropy Loss
- **Seed:** 42 (globally set for reproducibility)

### Experimental Setup
* **Centralized Baselines:** Trained for 15 epochs on the full training dataset.
* **Federated Learning:** 3 clients, 5 communication rounds, and 3 local epochs per round.
  * **Aggregation Algorithm:** Federated Averaging (`FedAvg`), where client weights are averaged proportionally to their local sample counts.
  * **FL IID Partitioning:** Shuffled training set split equally among 3 clients:
    - **Client 1:** 1,867 samples
    - **Client 2:** 1,867 samples
    - **Client 3:** 1,866 samples
  * **FL Non-IID Partitioning:** Heterogeneous split where 70% of each class's training images are assigned to one dominant client and the remaining 30% are split equally among the other 2 clients:
    - **Client 1:** 2,380 samples | `{'glioma': 979, 'meningioma': 211, 'notumor': 211, 'pituitary': 979}`
    - **Client 2:** 1,611 samples | `{'glioma': 211, 'meningioma': 979, 'notumor': 210, 'pituitary': 211}`
    - **Client 3:** 1,609 samples | `{'glioma': 210, 'meningioma': 210, 'notumor': 979, 'pituitary': 210}`
  * **Communication Cost:** Bidirectional parameter exchange (upload + download) per client:
    - **Per-Round Transfer Cost:** 342.7 MB
    - **Cumulative 5-Round Cost:** 1,713.5 MB

---

## Evaluation Metrics
- **Overall Accuracy:** Percentage of correct predictions on the 1,600 test images.
- **Precision, Recall, F1-Score:** Detailed per-class metrics to evaluate class-wise bias.
- **Confusion Matrix:** Map showing true vs. predicted counts for all classes.
- **Grad-CAM Visualization:** Saliency mapping on `layer4[-1]` outputs to inspect spatial focus areas.
- **Cumulative Communication Cost:** Tracking data transferred (in MB) across communication rounds.

---

## Actual Experimental Results

The models were evaluated on the 1,600 balanced test images. The results are summarized below:

### Overall Performance Comparison Table

| Method | Test Accuracy | Glioma F1 | Meningioma F1 | NoTumor F1 | Pituitary F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** | 95.12% | 0.909 | 0.931 | 0.973 | 0.989 |
| **Federated IID** | **95.19%** | **0.920** | **0.955** | 0.940 | **0.991** |
| **Federated Non-IID** | 94.12% | 0.903 | 0.931 | **0.950** | 0.977 |

*Note: The Centralized test accuracy at the final epoch was 95.12% (best epoch 14 reached 95.31%). The Federated models are loaded with their best-performing round weights.*

### Key Observations
- **Privacy-Performance Trade-off:**
  - **Centralized $\to$ FL IID Gap:** **-0.06%** (Federated IID matches and slightly exceeds the final epoch centralized baseline).
  - **FL IID $\to$ FL Non-IID Gap:** **1.06%** (Shows a minor performance drop due to class imbalance/heterogeneity across clients, proving FL robustness).

---

## How to Run the Notebook
The notebook is designed to run in **Google Colab** with GPU acceleration.

1. **Setup Google Drive:** Store the dataset folder named `Nickparvar` at `/content/drive/MyDrive/Nickparvar`.
2. **Library Installation:** Run the first code cell to install the required Grad-CAM library:
   ```bash
   pip install grad-cam -q
   ```
3. **Mount Google Drive:** Execute the drive mount cell and authorize access.
4. **Run Cells Sequentially:** Execute the script cell-by-cell. Training results and visual metrics will save automatically to `/content/drive/MyDrive/BrainTumor_FL_Results_Final`.

---

## Requirements
To execute locally or on Colab, ensure you have the libraries specified in `requirements.txt`. GPU acceleration (CUDA) is highly recommended for training.

---

## Project Structure

### Repository Structure
```
├── requirements.txt
├── README.md
└── src/
    └── Brain_Tumor_Final_Notebook.ipynb
```

### Output Directory Structure
Results are saved to Google Drive under `BrainTumor_FL_Results_Final/`:
```
├── Models/
│   ├── centralized_best_model.pth      # Best checkpoint of centralized training
│   ├── centralized_model.pth           # Last epoch centralized weights
│   ├── fl_iid_model.pth                # Best round weights of Federated IID
│   └── fl_noniid_model.pth             # Best round weights of Federated Non-IID
├── Accuracy & Metrics/
│   ├── central_accuracy.png            # Test accuracy curve for Centralized
│   ├── fl_iid_accuracy.png             # Global test accuracy curve for IID
│   ├── fl_noniid_accuracy.png          # Global test accuracy curve for Non-IID
│   └── comparison_accuracy.png         # Combined accuracy plot
├── Confusion Matrices/
│   ├── central_confusion_matrix.png
│   ├── fl_iid_confusion_matrix.png
│   └── fl_noniid_confusion_matrix.png
├── Cost/
│   └── communication_cost.png          # Cumulative bandwidth cost per round
└── Interpretability/
    └── gradcam_visualization.png       # Grad-CAM overlay comparison
```

---

## Limitations
- **High Communication Cost:** Exchanging updates for ~14.97M parameters per client per round requires 342.7 MB of data transfer per round (totaling 1.7 GB for 5 rounds), indicating a need for model compression.
- **Limited Simulation Scale:** Only 3 clients and 5 rounds are simulated, which does not fully capture complex, real-world decentralized clinicial distributions.
- **Environment Coupling:** The code contains hardcoded paths to Google Drive (`/content/drive/MyDrive/`) and utilizes Colab imports, meaning paths need adjustment to run in a standalone local directory.

---

## References
- **Nickparvar Dataset:** The source dataset of brain tumor MRI images.