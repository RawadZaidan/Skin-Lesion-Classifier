"""
Skin Lesion Classification for Dermatological AI
=================================================
This project explores machine learning approaches to classifying skin lesions
using the HAM10000 dataset metadata. It is motivated by the challenge of building
diagnostic AI tools that work under real-world clinical constraints — limited data,
class imbalance, and the need for interpretable, clinically meaningful outputs.

Dataset: HAM10000 (Human Against Machine with 10000 training images)
         Metadata only — no image download required.
         Source: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

Relevant research context:
- Tegnér Lab (KAUST): Machine learning for biological and medical imaging,
  including melanoma datasets and retinal disease
- Gao Lab (KAUST): Biomedical imaging analysis, omics-based disease detection

Author: Rawad Zeidan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             balanced_accuracy_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─── 1. SIMULATE HAM10000 METADATA ────────────────────────────────────────────
# In a full run, load the real CSV from Kaggle:
#   df = pd.read_csv('HAM10000_metadata.csv')
# Here we reproduce the statistical distribution of the real dataset
# so the pipeline logic is identical.

np.random.seed(42)
N = 10015  # actual HAM10000 size

# Diagnosis distribution (reflects true HAM10000 class imbalance)
dx_classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
dx_probs   = [0.669, 0.111, 0.110, 0.051, 0.033, 0.014, 0.012]
dx         = np.random.choice(dx_classes, size=N, p=dx_probs)

# Localization
locs = ['back', 'lower extremity', 'trunk', 'upper extremity',
        'abdomen', 'face', 'chest', 'foot', 'scalp', 'neck', 'hand', 'ear', 'genital']
loc_probs = [0.20, 0.18, 0.15, 0.12, 0.08, 0.07, 0.06, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01]
localization = np.random.choice(locs, size=N, p=loc_probs)

# Age (realistic distribution, some missing)
age_base = np.random.normal(52, 16, N).clip(5, 85)
age_base[np.random.rand(N) < 0.1] = np.nan  # ~10% missing

# Sex
sex = np.random.choice(['male', 'female', 'unknown'], size=N, p=[0.53, 0.46, 0.01])

# Confirmation method
dx_type = np.random.choice(['histo', 'follow_up', 'consensus', 'confocal'],
                            size=N, p=[0.54, 0.26, 0.18, 0.02])

df = pd.DataFrame({
    'lesion_id': [f'HAM_{i:05d}' for i in range(N)],
    'dx': dx,
    'dx_type': dx_type,
    'age': age_base,
    'sex': sex,
    'localization': localization
})

print("=" * 60)
print("SKIN LESION CLASSIFICATION — HAM10000 Metadata Pipeline")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['dx'].value_counts())
print(f"\nClass imbalance ratio (majority/minority): "
      f"{df['dx'].value_counts().max() / df['dx'].value_counts().min():.1f}x")

# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n--- Feature Engineering ---")

df['age_filled'] = df['age'].fillna(df['age'].median())
df['age_missing'] = df['age'].isna().astype(int)

# Body region grouping (clinically motivated)
high_risk_locs = ['face', 'ear', 'scalp', 'neck', 'genital']
df['high_risk_location'] = df['localization'].isin(high_risk_locs).astype(int)

# Encode categoricals
le_sex = LabelEncoder()
le_loc = LabelEncoder()
le_dxtype = LabelEncoder()

df['sex_enc']    = le_sex.fit_transform(df['sex'])
df['loc_enc']    = le_loc.fit_transform(df['localization'])
df['dxtype_enc'] = le_dxtype.fit_transform(df['dx_type'])

# Binary target: melanoma (mel) vs. all others — clinically the highest-stakes decision
df['target'] = (df['dx'] == 'mel').astype(int)
print(f"Melanoma cases: {df['target'].sum()} / {len(df)} "
      f"({df['target'].mean()*100:.1f}%)")

features = ['age_filled', 'age_missing', 'sex_enc', 'loc_enc',
            'dxtype_enc', 'high_risk_location']
X = df[features].values
y = df['target'].values

# ─── 3. TRAIN/TEST SPLIT + SMOTE ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Training set: {X_train_bal.shape[0]} samples "
      f"(was {X_train.shape[0]})")

# ─── 4. MODELS ────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

print("\n--- Model Evaluation (clinically relevant metrics) ---")
print(f"{'Model':<25} {'AUC':>6} {'Sensitivity':>12} {'Specificity':>12} {'Bal.Acc':>9}")
print("-" * 70)

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train_bal)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'auc': auc, 'sensitivity': sensitivity,
        'specificity': specificity, 'bal_acc': bal_acc
    }
    print(f"{name:<25} {auc:>6.3f} {sensitivity:>12.3f} {specificity:>12.3f} {bal_acc:>9.3f}")

# ─── 5. VISUALISATIONS ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Skin Lesion Classification — Model Evaluation", fontsize=14, fontweight='bold')

# ROC curves
ax = axes[0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curves")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Sensitivity vs Specificity bar chart
ax = axes[1]
names = list(results.keys())
sens  = [results[n]['sensitivity'] for n in names]
spec  = [results[n]['specificity'] for n in names]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, sens, w, label='Sensitivity', color='#c0392b', alpha=0.8)
ax.bar(x + w/2, spec, w, label='Specificity', color='#2980b9', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Clinical Metrics\n(Sensitivity = true melanoma detection rate)")
ax.legend()
ax.set_ylim(0, 1.05)
ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
ax.grid(axis='y', alpha=0.3)

# Class distribution
ax = axes[2]
class_counts = df['dx'].value_counts()
class_labels = {
    'nv': 'Melanocytic nevi', 'mel': 'Melanoma',
    'bkl': 'Benign keratosis', 'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratosis', 'vasc': 'Vascular lesion', 'df': 'Dermatofibroma'
}
colors = ['#e74c3c' if c == 'mel' else '#3498db' for c in class_counts.index]
ax.bar([class_labels.get(c, c) for c in class_counts.index],
       class_counts.values, color=colors, alpha=0.8)
ax.set_title("HAM10000 Class Distribution\n(red = melanoma — the clinical priority)")
ax.set_ylabel("Count")
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: results.png")

# ─── 6. FEATURE IMPORTANCE ────────────────────────────────────────────────────
rf = results["Random Forest"]['model']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Random Forest Feature Importances ---")
for feat, imp in importances.items():
    print(f"  {feat:<25} {imp:.4f}")

print("\n--- Key Clinical Findings ---")
best = max(results, key=lambda n: results[n]['auc'])
r = results[best]
print(f"  Best model: {best}")
print(f"  AUC:         {r['auc']:.3f}")
print(f"  Sensitivity: {r['sensitivity']:.3f}  (of every 100 melanomas, ~{r['sensitivity']*100:.0f} detected)")
print(f"  Specificity: {r['specificity']:.3f}  (of every 100 benign, ~{r['specificity']*100:.0f} correctly cleared)")
print("\nNote: In clinical screening, high sensitivity is prioritised over")
print("specificity — missing a melanoma (false negative) carries greater")
print("harm than a false alarm (false positive) that triggers biopsy.")
print("\nPipeline complete.")
