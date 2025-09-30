#  Wine Dataset Classification with LinearSVC

This project implements a **LinearSVC classifier** on the **Wine dataset** from scikit-learn.  
It demonstrates a clean ML workflow using **Pipelines**, preprocessing, and robust evaluation techniques.

---

Results:

* Test Accuracy:** 97.2%  
* Cross-Validation:** Mean = 98.5%, Std = 0.017 (5-fold StratifiedKFold)  
* Dataset:** Wine (178 samples, 13 features, 3 classes)  
* Train-Test Split:** 80/20  



# Key Features
- ✅ Split the dataset into training (80%) and testing (20%) using train_test_split.  
- ✅ Used **Polynomial Features** (degree=2) to capture non-linear relationships.  
- ✅ Standardized features using **StandardScaler**.  
- ✅ Built a clean **Pipeline** combining preprocessing and LinearSVC training.  
- ✅ Applied **StratifiedKFold** cross-validation (5 splits) for robust performance evaluation.  
- ✅ Achieved **mean cross-validation score of 98.5%** with standard deviation 0.017.  
- ✅ Fitted the model on training data and predicted the test set.  
- ✅ Computed and displayed the **Confusion Matrix** for test predictions.  

---

## ⚙️ Tech Stack
- Python  
- scikit-learn  
- NumPy  
- Matplotlib  

---

## 📂 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/wine-linearSVC.git
