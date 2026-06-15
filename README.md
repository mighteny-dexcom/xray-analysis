# XRay Analysis – Battery Tab Defect Detection (Deep Learning Pipeline)

This project implements a **deep learning–based screening system** for detecting battery tab (BT / pBT) defects in X-ray images using **transfer learning (ResNet18)** and **ROI-based modeling**.

---

## 📁 Project Structure

```text
xray-analysis/
├── train_xray_v2_screening.py           # holdout-lot experiments
├── train_xray_v3_all_lots_split.py      # all-lot stratified training
├── train_xray_v3_all_lots_split_roi.py  # ROI-enhanced training (best model)
├── evaluate_thresholds.py               # threshold sweep
├── infer_lot4.py                        # inference on new lots
├── models/
├── lot*_images/
└── README.md
---
Model Development Timeline
1) Initial Transfer Learning (No Validation)
Trained ResNet18 on all lots
Only tracked training loss
No validation → no insight into generalization
Lesson: Validation is required.
---
2) Leave-One-Lot-Out Validation
Train on two lots, validate on the third:
Train: Lot2 + Lot3 → Val: Lot1
Train: Lot1 + Lot3 → Val: Lot2
Train: Lot1 + Lot2 → Val: Lot3
Findings
Lot2/3 strong
Lot1 hardest
Observed "bad misses" (false negatives)
---
3) All-Lot Training with Stratified Split
Per lot:
80% train
20% validation (stratified)
Combined:
Train on all lots
Validate on held-out subset
Results (Full Image Model)
Sensitivity: 1.000
Specificity: ~0.90
FN: 0
FP: ~10
---
4) ROI-Based Training (Major Improvement)
Added fixed ROI crop:
```python
ROI_BOX = (450, 350, 650, 500)
img = img.crop(ROI_BOX)
```
Results (ROI Model)
Model	FN	FP	Sens	Spec
Full Image	0	10	1.000	0.902
ROI Model	0	3	1.000	0.971
Impact
Same detection performance
~70% fewer false positives
---
Final Model Configuration
Model:
```
best_screening_resnet18_all_lots_split_roi.pth
```
Threshold:
```python
THRESHOLD = 0.40
```
ROI:
```python
ROI_BOX = (450, 350, 650, 500)
```
---
Inference on New Data (Lot4)
Run:
```
python infer_lot4.py
```
Outputs:
```
filename | probability | predicted_label
```
---
Review Workflow
Review flagged images (pred = 1)
Sort by probability
Inspect borderline cases (0.35–0.50)
Spot-check negatives
---
Key Insights
Validation strategy matters more than model choice
Threshold controls FN vs FP tradeoff
ROI dramatically improves performance for localized defects
Data diversity remains the main limitation
---
Future Work
Add more defect examples
Multi-ROI for multiple defect types
Object detection models
Validation on new lots (Lot4+)
---
Final Takeaway
A robust inspection system requires:
Proper validation strategy
Threshold tuning
Spatial focus (ROI)
