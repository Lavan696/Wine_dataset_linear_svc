# Wine Data Prediction using Polynomial Linear SVM

## Overview  
This project focuses on classifying different wine cultivars using the **Wine dataset** from `scikit-learn`.  
The model utilizes a **Polynomial + Linear SVM** pipeline with **calibrated probabilities** for robust multi-class classification.

---

## Workflow  
1. **Data Loading** – Loaded the Wine dataset using `load_wine()` from sklearn.  
2. **EDA** – Checked correlations, visualized feature relationships, and analyzed target distribution.  
3. **Preprocessing** – Standardized features and added polynomial terms for non-linear separability.  
4. **Model Training** – Trained a **Polynomial Linear SVM** wrapped in a **CalibratedClassifierCV** for probability estimates.  
5. **Evaluation** – Measured multiple metrics to validate model performance on test data.  
6. **Visualization** – Created correlation heatmaps, ROC & Precision-Recall curves, confusion matrix, and classification heatmaps.  

---

## Model Evaluation  

| Metric                                 | Score              |
|:---------------------------------------|:-------------------|
| **Cross-Val Mean Acc ± Std Deviation** | **99.28% ± 0.021** |
| **Test Accuracy**                      | **97.22%**         |
| **Precision (Weighted)**               | **97.40%**         |
| **Recall (Weighted)**                  | **97.22%**         |
| **F1-Score (Weighted)**                | **97.22%**         |
| **ROC-AUC (OVR)**                      | **99.89%**         |
| **Log Loss**                           | **0.231**          |
| **Cohen’s Kappa Score**                | **95.64%**         |
| **Matthews Corrcoef (MCC)**            | **95.76%**         |
| **Top-K Accuracy (k=2)**               | **100%**           |

---

##  Visual Insights  
-  **Feature Correlation Heatmap** – Showed which features strongly influence target classes.  
-  **ROC and Precision-Recall Curves** – Demonstrated near-perfect separability among classes.  
-  **Confusion Matrix** – Displayed minimal misclassifications.  
-  **Classification Report Heatmap** – Summarized per-class precision, recall, and F1 visually.  

---

##  Key Insights  
- Polynomial feature expansion enhanced non-linear decision boundaries significantly.  
- Calibrating LinearSVC allowed reliable **probability estimates** for metrics like ROC-AUC and Log Loss.  
- Consistent performance across folds (low std deviation) shows **high model stability**.  
- Excellent agreement metrics (Cohen’s Kappa & MCC ≈ 95%) confirm **robust multi-class consistency**.

---

## Tech Stack  
- **Python Libraries:** NumPy, Pandas, Matplotlib, Seaborn  
- **ML Frameworks:** Scikit-learn  
- **Model Used:** PolynomialFeatures + StandardScaler + Calibrated LinearSVC
  
---

##  Model Saving  
The trained model is serialized using `joblib` for easy deployment:  
`python
joblib.dump(poly_lin_svm_pipeline, 'wine_poly_svm.pkl')`

---

##  Author  

**Lavan Kumar Konda**  
-  2nd Year Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)

