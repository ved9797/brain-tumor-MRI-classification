# Brain Tumor MRI Image Classification — Submission Package

## Contents
- `Brain_Tumor_MRI_Classification_Final.ipynb`: cleaned notebook with EDA, model training/evaluation, and deployment artifact export.
- `app.py`: Streamlit inference demo.
- `requirements.txt`: application dependencies.
- `brain_tumor_mobilenetv2.keras`: the uploaded trained MobileNetV2 model.
- `class_names.json`: the class order used during training.
- `evaluation_summary.json`, `classification_report.csv`, `confusion_matrix.csv`: the reported held-out test results.

## Run in Google Colab
1. Upload the notebook and the train/valid/test ZIP archives.
2. Run all cells from top to bottom. Training is enabled in the configuration cell.
3. The final export cell writes artifacts to `/content/brain_tumor_submission`.
4. Download the `.keras` model, `class_names.json`, and evaluation files. Colab storage is temporary.

## Run the Streamlit app locally
Place `app.py`, `requirements.txt`, `brain_tumor_mobilenetv2.keras`, and `class_names.json` in one directory, then run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The model expects RGB image pixels in the 0–255 range because preprocessing is embedded in the saved MobileNetV2 model. The class mapping is loaded from `class_names.json` to preserve the training label order.

## Current reported evaluation
The reported held-out test accuracy for MobileNetV2 is 87.80% on 246 images (216/246 correct). Reported macro F1 is 0.87. Meningioma recall is 68.25%, so this class is a known weakness. These metrics are based on the supplied evaluation output; rerun evaluation to independently verify the uploaded model file.

## Medical-use limitation
This is an educational prototype, not a clinically validated medical device. It must not be used to diagnose, rule out disease, or make treatment decisions. Evaluation on an independent, representative dataset and qualified clinical review would be required before any real-world use.
