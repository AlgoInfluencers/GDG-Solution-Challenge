import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import warnings

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic data that contains intentional historical bias
    against Group 0 (the disadvantaged/minority group).
    """
    X_syn, y_syn = make_classification(n_samples=n_samples, n_features=10, n_informative=5, random_state=42)
    
    # Create sensitive attribute 'Gender' (0: e.g., Female/Minority, 1: e.g., Male/Majority)
    A_syn = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    # Introduce bias manually:
    # If A=0 (minority), flip positive outcomes (y=1) to 0 with 50% probability.
    # This means qualified individuals in Group 0 are artificially rejected more often.
    y_biased = y_syn.copy()
    flip_indices = (A_syn == 0) & (y_biased == 1) & (np.random.rand(n_samples) < 0.5)
    y_biased[flip_indices] = 0
    
    feature_cols = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X_syn, columns=feature_cols)
    df['Gender'] = A_syn
    df['Target'] = y_biased
    
    return df

def calculate_metrics(y_true, y_pred, A):
    """Calculate Performance and Fairness Metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    mask_A0 = (A == 0)
    mask_A1 = (A == 1)
    
    # Statistical Parity Difference (SPD) = P(Y^=1 | A=0) - P(Y^=1 | A=1)
    p_y1_A0 = np.mean(y_pred[mask_A0] == 1) if sum(mask_A0) > 0 else 0
    p_y1_A1 = np.mean(y_pred[mask_A1] == 1) if sum(mask_A1) > 0 else 0
    spd = p_y1_A0 - p_y1_A1
    
    # Equal Opportunity Difference (EOD) = TPR_A0 - TPR_A1
    # TPR = True Positives / Actual Positives
    tpr_A0 = np.mean(y_pred[mask_A0 & (y_true == 1)] == 1) if sum(mask_A0 & (y_true == 1)) > 0 else 0
    tpr_A1 = np.mean(y_pred[mask_A1 & (y_true == 1)] == 1) if sum(mask_A1 & (y_true == 1)) > 0 else 0
    eod = tpr_A0 - tpr_A1
    
    return accuracy, f1, spd, eod, mask_A0, mask_A1

def print_confusion_matrices(y_true, y_pred, mask_A0, mask_A1):
    """Prints a clear confusion matrix per sensitive group."""
    print(f"\n--- Confusion Matrix per Group ---")
    cm_0 = confusion_matrix(y_true[mask_A0], y_pred[mask_A0], labels=[0,1])
    cm_1 = confusion_matrix(y_true[mask_A1], y_pred[mask_A1], labels=[0,1])
    
    print("Group 0 (Disadvantaged):")
    print(f"[[TN={cm_0[0,0]} FP={cm_0[0,1]}]\n [FN={cm_0[1,0]} TP={cm_0[1,1]}]]")
    print("\nGroup 1 (Advantaged):")
    print(f"[[TN={cm_1[0,0]} FP={cm_1[0,1]}]\n [FN={cm_1[1,0]} TP={cm_1[1,1]}]]")

def train_and_eval(X_train, y_train, X_test, y_test, A_test, sample_weight=None, model_name=""):
    """Modular function to train a model and evaluate metrics."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    accuracy, f1, spd, eod, mask_A0, mask_A1 = calculate_metrics(y_test, y_pred, A_test)
    
    print(f"\n================ {model_name.upper()} ================")
    print(f"Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
    
    # Add Threshold check: ideally, SPD and EOD should be close to 0 (between -0.1 and 0.1)
    spd_status = "✅" if abs(spd) <= 0.1 else "❌"
    eod_status = "✅" if abs(eod) <= 0.1 else "❌"
    
    print(f"SPD: {spd:.4f} {spd_status}")
    print(f"EOD: {eod:.4f} {eod_status}")
    
    if abs(spd) > 0.1:
        print("⚠ Bias Detected: Statistical Parity Difference exceeds the allowed limit (0.1)")
        
    print_confusion_matrices(y_test, y_pred, mask_A0, mask_A1)
    
    return model

def compute_reweights(A, y):
    """
    Computes class weights based on the joint distribution of the sensitive attribute 
    and target variable. Formula: W = P(A) * P(Y) / P(A, Y)
    """
    df = pd.DataFrame({'A': A, 'y': y})
    n = len(df)
    
    p_A = df['A'].value_counts() / n
    p_y = df['y'].value_counts() / n
    p_Ay = df.groupby(['A', 'y']).size() / n
    
    weights = np.zeros(n)
    for i in range(n):
        a_val, y_val = df.iloc[i]['A'], df.iloc[i]['y']
        weights[i] = (p_A[a_val] * p_y[y_val]) / p_Ay.loc[(a_val, y_val)]
        
    return weights

def main():
    parser = argparse.ArgumentParser(description="Bias Detection and Mitigation ML Pipeline")
    parser.add_argument('--csv', type=str, default=None, help='(Optional) Path to real CSV dataset')
    parser.add_argument('--target_col', type=str, default='Target', help='Target column name')
    parser.add_argument('--sensitive_col', type=str, default='Gender', help='Sensitive attribute column (0/1 format)')
    args = parser.parse_args()
    
    print("🧠 Starting ML Bias Detection & Mitigation Pipeline 🧠\n")
    
    if args.csv:
        print(f"Loading real data from {args.csv}...")
        df = pd.read_csv(args.csv)
        
        # Data Preprocessing: Fill missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Label Encoding for categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            print(f"Encoding categorical column: {col}")
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    else:
        print("💡 No CSV provided. Generating Synthetic Dataset (Controlled Bias Simulation)...")
        df = generate_synthetic_data()

    target_col = args.target_col
    sensitive_col = args.sensitive_col
    
    if target_col not in df.columns or sensitive_col not in df.columns:
        raise ValueError(f"Columns '{target_col}' or '{sensitive_col}' not found. Check your column names.")
        
    print("\n🟢 Step 1: Preprocessing & Defining Variables")
    feature_cols = [c for c in df.columns if c not in [target_col, sensitive_col]]
    
    X = df[feature_cols + [sensitive_col]].copy()
    y = df[target_col].copy()
    
    # Scale Features
    scaler = StandardScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extract sensitive attribute for metric calculation
    A_train = X_train[sensitive_col].values
    A_test = X_test[sensitive_col].values

    print("🔵 Step 2 & 3: Training Baseline ML Model & Detecting Bias")
    # Base model evaluates everything with the sensitive attribute still in the dataset.
    train_and_eval(X_train, y_train, X_test, y_test, A_test, model_name="BEFORE MITIGATION")
    
    print("\n\n🛠️  Step 4: APPLYING BIAS MITIGATION TECHNIQUES 🛠️")
    
    # -------------------------------------------------------------
    # MITIGATION 1: Feature Removal
    # -------------------------------------------------------------
    print("\n>>> Technique 1: FEATURE REMOVAL (Dropping sensitive attribute)")
    X_train_fr = X_train.drop(columns=[sensitive_col])
    X_test_fr = X_test.drop(columns=[sensitive_col])
    train_and_eval(X_train_fr, y_train, X_test_fr, y_test, A_test, model_name="AFTER MITIGATION (FEATURE REMOVAL)")
    
    # -------------------------------------------------------------
    # MITIGATION 2: Reweighting
    # -------------------------------------------------------------
    print("\n>>> Technique 2: REWEIGHTING (Give higher importance to disadvantaged groups)")
    sample_weights = compute_reweights(A_train, y_train.values)
    train_and_eval(X_train, y_train, X_test, y_test, A_test, sample_weight=sample_weights, model_name="AFTER MITIGATION (REWEIGHTING)")
    
    # -------------------------------------------------------------
    # MITIGATION 3: Re-sampling
    # -------------------------------------------------------------
    print("\n>>> Technique 3: RE-SAMPLING (Balancing dataset equal groups)")
    # To re-sample properly, we want to balance the combination of sensitive attribute AND target.
    # We combine them into a single String class so the RandomUnderSampler balances all 4 combinations (0_0, 0_1, 1_0, 1_1) equally.
    y_composite = X_train[sensitive_col].astype(str) + "_" + y_train.astype(str)
    
    rus = RandomUnderSampler(random_state=42)
    # Re-sample based on the composite group logic
    X_resampled, y_resampled_comp = rus.fit_resample(X_train, y_composite)
    
    # Extract original 'y' from composite mapping
    y_resampled = y_resampled_comp.apply(lambda val: int(val.split("_")[1]))
    
    train_and_eval(X_resampled, y_resampled, X_test, y_test, A_test, model_name="AFTER MITIGATION (RE-SAMPLING)")

if __name__ == "__main__":
    main()
