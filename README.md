# Blood Cell Image Classification (CNN)


A clean, end‑to‑end Convolutional Neural Network (CNN) pipeline to classify **normal peripheral blood cells** from microscopy images.  
The project is lightweight, Colab‑friendly, and built to be **easy to read, run, and extend**.

---

## Highlights
- **Plug‑and‑play dataset** via `kagglehub` (fast Colab cache).
- **Reproducible notebook**: `Blood_Cell_Classification.ipynb` contains the full pipeline (data → transforms → training → evaluation).
- **Clear structure & styling** for quick understanding and easy reporting.
- **Metrics‑ready**: accuracy, per‑class performance, confusion matrix & sample predictions (cells to add in the notebook).

---

## Project Structure
```
    blood-cell-cnn/
├── Blood_Cell_Classification.ipynb   # Main notebook (train + evaluate + visualize)
├── README.md                         # This file

```
> The dataset is automatically fetched into Colab’s cache with `kagglehub`.  
> If running locally, you can set a custom `data/` path.

---

## Dataset
- **Source**: Kaggle — *Blood Cells Image Dataset*  
  https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset
- **Image size**: `360 × 363` JPG
- **Classes (8):** `neutrophils`, `eosinophils`, `basophils`, `lymphocytes`, `monocytes`, `immature granulocytes` (promyelocyte/myelocyte/metamyelocyte), `erythroblasts`, `platelets`

>  The images were annotated by expert clinical pathologists and captured from healthy individuals.

---

## Quickstart

### 1) Run on Google Colab (recommended)
Open the notebook in Colab and run all cells. The dataset will be downloaded and cached automatically.

```python
!pip -q install kagglehub pillow torch torchvision

import kagglehub, os, glob
from PIL import Image

# Download latest version
path = kagglehub.dataset_download("unclesamulus/blood-cells-image-dataset")
root = os.path.join(path, "bloodcells_dataset")

# Peek one image
one_image = glob.glob(os.path.join(root, "basophil", "*.jpg"))[0]
Image.open(one_image).convert("RGB")
```


---

## Model
- **Backbone**: A compact CNN (conv → ReLU → maxpool → dropout) → FC classifier.
- **Transforms**: Resize/CenterCrop, normalization;  light augmentations.
- **Loss / Optimizer**: `CrossEntropyLoss` + SGD/Adam.
- **Evaluation**: overall accuracy, class‑wise metrics, confusion matrix.

> You can easily swap in a pretrained backbone (e.g., `torchvision.models.resnet18`) for transfer learning.

---

## Training & Evaluation (inside the notebook)
- Configure hyperparameters (epochs, lr, batch size).
- Train the model with live loss/accuracy.
- Save artifacts: trained weights (`.pt`), label map, plots.
- Inspect predictions on a validation/test subset.

**Tip:** For small datasets, prefer **stratified splits** and enable **early stopping** to reduce overfitting.

---

##  Results Summary

- CNN Acc: 0.9657 | ResNet50 Acc: 0.9750

---

