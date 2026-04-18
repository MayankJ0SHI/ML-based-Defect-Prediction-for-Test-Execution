# 🚀 ML-based Defect Prediction for Test Execution

## 📌 Overview
This project uses Machine Learning techniques to predict defect-prone areas in software systems, helping optimize test execution by prioritizing high-risk components.

The goal is to move from executing all test cases to a **risk-based testing approach**, improving efficiency and defect detection.

---

## 🎯 Objectives
- Predict whether a module/test case is defect-prone  
- Improve test prioritization  
- Reduce overall testing effort and execution time  
- Support smarter QA decision-making  

---

## 🧠 Approach

The project follows a standard Machine Learning pipeline:

1. **Data Collection**
   - Historical defect/test execution data  

2. **Data Preprocessing**
   - Handling missing values  
   - Encoding categorical features  
   - Feature selection  

3. **Model Training**
   - Train classification models to predict defects  

4. **Model Evaluation**
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  

5. **Prediction**
   - Identify high-risk areas for targeted testing  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/MayankJ0SHI/ML-based-Defect-Prediction-for-Test-Execution.git

# Navigate to project directory
cd ML-based-Defect-Prediction-for-Test-Execution

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
