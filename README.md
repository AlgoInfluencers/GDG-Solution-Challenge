# Fair ML Platform (GDG Solution Challenge)

An end-to-end Bias Detection, Mitigation, and Structural Network Analysis pipeline to ensure ethical and fair Machine Learning models.

## 🚀 Overview
Machine learning models often inherit historical biases based on attributes like Gender, Age, or Race. This project tackles this problem by identifying instances of bias, measuring it mathematically, completely mitigating it using various algorithmic strategies, and mapping out the structural isolation of demographic groups using graph theory.

This application consists of:
1. **Python ML Pipeline (`ml_pipeline.py`)**: Automatically evaluates "Statistical Parity Difference" and "Equal Opportunity Difference," and applies mitigation strategies (Feature Removal, Reweighting, Re-sampling).
2. **C++ Analytics Engine (`graph.cpp`)**: A lightning-fast script that constructs graphical network similarities entirely out of tabular rows.
3. **Interactive Dashboard (`app.py`)**: A stunning Streamlit application bridging the backend to the user. 

## 🛠️ Installation
```bash
# Clone the repository
git clone <your-repo>

# Setup Python Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt

# Compile C++ backend (if not precompiled)
g++ graph.cpp -o g.exe
```

## 💻 Usage

To run the fully integrated Streamlit dashboard UI:
```bash
streamlit run app.py
```
This will open `localhost:8501`. From here, you can upload a unique CSV, select the specific Target and Sensitive Demographic Columns, and hit run!

To run blindly through the CLI:
```bash
python ml_pipeline.py --csv data.csv --target_col Loan_Status --sensitive_col Gender
```

## 📊 Analytics
Once executed, the platform successfully tests Fairness vs. Accuracy and mitigates bias.
- **Fairness Guarantee**: The dashboard produces `fair_model.joblib` containing the serialized fairest model for direct deployment API integration.
- **Visual Tracking**: Generates `metrics_tradeoff.png` showing the mitigation journey.
- **Network Bias**: Generates `structural_bias.png` visually highlighting marginalized societal bubbles within the dataset structure.

---
Built for the **Google Developer Groups (GDG) Solution Challenge**.
