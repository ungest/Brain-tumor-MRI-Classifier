# Brain Tumor MRI Classification

This project focuses on classifying brain tumors into three categories — **Meningioma**, **Glioma**, and **Pituitary** — using MRI scans. We leverage CNN-based architectures including **VGG16**, **ConvNeXt-Small**, and **ViT** for comparative performance analysis. The app interface is built using **Streamlit**.

## Dataset

The dataset used is the publicly available [brain-tumor-mri-dataset](https://github.com/guillaumefrd/brain-tumor-mri-dataset), originally sourced from figshare.

* MRI images are stored in `.mat` files with pixel data and class labels
* Images were converted to grayscale `.png` files for training and evaluation

## Notebooks

1. **EDA\_on\_dataset.ipynb**

   * Visualizes class distribution
   * Displays random samples
   * Highlights grayscale nature of input
   * Includes a simple baseline CNN model for quick benchmarking

2. **Second\_model\_2.ipynb**

   * Contains model development for VGG16, ViT, and ConvNeXt
   * Includes performance evaluation using accuracy, F1-score, confusion matrix, and ROC curves

## Models Used

* **VGG16** (baseline)
* **ConvNeXt-Small** (best performer)
* **ViT** (Vision Transformer)

All models were fine-tuned with the last classification layer adjusted to output 3 classes.

## Streamlit App

* Uploads single MRI images and predicts tumor type
* Displays class-wise prediction confidence with charts
* Handles grayscale → RGB conversion and ImageNet normalization

## 🗃️ Project Structure

```
brain_tumor_dataset/
├── brain_mri_app/
│   └── streamlit_app.py
├── Notebooks/
│   ├── EDA_and_Baseline_Model_notebook.ipynb
│   └── Models_Development_and_Evaluation.ipynb
├── EDA_on_dataset.ipynb
├── Second_model_2.ipynb
└── README.md
└── requirements.txt

```

## 📈 Sample Results

| Model    | Accuracy  | Macro F1 | Pituitary F1 | Meningioma F1 | Glioma F1 |
| -------- | --------- | -------- | ------------ | ------------- | --------- |
| VGG16    | 92.3%     | 0.91     | 0.95         | 0.89          | 0.89      |
| ConvNeXt | **95.1%** | **0.94** | **0.96**     | **0.93**      | **0.94**  |
| ViT      | 90.8%     | 0.89     | 0.93         | 0.87          | 0.86      |

## 🔗 Reference

Dataset and original source: [https://github.com/guillaumefrd/brain-tumor-mri-dataset](https://github.com/guillaumefrd/brain-tumor-mri-dataset)

---

For questions or contributions, feel free to open an issue or submit a PR.
