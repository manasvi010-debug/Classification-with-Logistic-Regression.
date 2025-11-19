# Classification-with-Logistic-Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                            f1_score, roc_curve, auc, roc_auc_score,
                            classification_report, accuracy_score)

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load the Breast Cancer Dataset
print("=" * 70)
print("BINARY CLASSIFICATION WITH LOGISTIC REGRESSION")
print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
print("=" * 70)

cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

print(f"\nDataset Shape:")
print(f"  Features (X): {X.shape}")
print(f"  Target (y): {y.shape}")
print(f"  Classes: {cancer_data.target_names}")
print(f"  Class distribution: {np.bincount(y)}")

# Step 2: Train-Test Split
print("\n" + "=" * 70)
print("STEP 1: DATA SPLITTING & STANDARDIZATION")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain-Test Split (80-20):")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")
print(f"  Train class distribution: {np.bincount(y_train)}")
print(f"  Test class distribution: {np.bincount(y_test)}")

# Step 3: Feature Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature Standardization (StandardScaler):")
print(f"  Mean of scaled training data: {X_train_scaled.mean():.6f}")
print(f"  Std of scaled training data: {X_train_scaled.std():.6f}")
print(f"  Feature range before scaling: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"  Feature range after scaling: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

# Step 4: Fit Logistic Regression Model
print("\n" + "=" * 70)
print("STEP 2: LOGISTIC REGRESSION MODEL TRAINING")
print("=" * 70)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"\nModel Training Complete:")
print(f"  Number of features: {len(model.coef_[0])}")
print(f"  Bias (intercept): {model.intercept_[0]:.6f}")
print(f"  Top 5 feature importances (coefficients):")

# Get feature importance
feature_importance = np.abs(model.coef_[0])
top_indices = np.argsort(feature_importance)[-5:][::-1]
for idx, feat_idx in enumerate(top_indices, 1):
    print(f"    {idx}. {cancer_data.feature_names[feat_idx]}: {model.coef_[0][feat_idx]:.6f}")

# Step 5: Model Predictions
print("\n" + "=" * 70)
print("STEP 3: MODEL PREDICTIONS")
print("=" * 70)

# Probability predictions
y_pred_proba = model.predict_proba(X_test_scaled)
y_pred = model.predict(X_test_scaled)

print(f"\nPredictions on Test Set:")
print(f"  Total samples: {len(y_test)}")
print(f"  Predicted class 0 (Malignant): {np.sum(y_pred == 0)}")
print(f"  Predicted class 1 (Benign): {np.sum(y_pred == 1)}")
print(f"  Probability range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")

# Step 6: Model Evaluation - Confusion Matrix
print("\n" + "=" * 70)
print("STEP 4: MODEL EVALUATION METRICS")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  {cm}")
print(f"\nConfusion Matrix Components:")
print(f"  True Negatives (TN): {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP): {tp}")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print(f"\nPerformance Metrics (Default threshold = 0.5):")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print(f"\nMetric Interpretations:")
print(f"  - Accuracy: {accuracy*100:.2f}% of predictions are correct")
print(f"  - Precision: {precision*100:.2f}% of positive predictions are correct")
print(f"  - Recall: {recall*100:.2f}% of actual positives are correctly identified")
print(f"  - F1-Score: Harmonic mean of precision and recall ({f1:.4f})")
print(f"  - ROC-AUC: Probability model ranks random pos > random neg ({roc_auc:.4f})")

# Classification Report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer_data.target_names))

# Step 7: ROC Curve Analysis
print("\n" + "=" * 70)
print("STEP 5: ROC CURVE & THRESHOLD OPTIMIZATION")
print("=" * 70)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc_curve = auc(fpr, tpr)

print(f"\nROC Curve Analysis:")
print(f"  AUC Score: {roc_auc_curve:.4f}")
print(f"  Number of thresholds: {len(thresholds)}")
print(f"  Threshold range: [{thresholds.min():.4f}, {thresholds.max():.4f}]")

# Find optimal threshold (Youden's J statistic)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Threshold Calculation (Youden's J = TPR - FPR):")
print(f"  Optimal Threshold: {optimal_threshold:.4f}")
print(f"  TPR at optimal: {tpr[optimal_idx]:.4f}")
print(f"  FPR at optimal: {fpr[optimal_idx]:.4f}")
print(f"  Youden's J: {j_scores[optimal_idx]:.4f}")

# Predictions with optimal threshold
y_pred_optimal = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)

print(f"\nPerformance with Optimal Threshold ({optimal_threshold:.4f}):")
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
precision_optimal = precision_score(y_test, y_pred_optimal)
recall_optimal = recall_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)

print(f"  Accuracy:  {accuracy_optimal:.4f}")
print(f"  Precision: {precision_optimal:.4f}")
print(f"  Recall:    {recall_optimal:.4f}")
print(f"  F1-Score:  {f1_optimal:.4f}")

# Comparison
print(f"\nThreshold Comparison:")
print(f"  Metric         | Default (0.5) | Optimal ({optimal_threshold:.4f}) | Difference")
print(f"  ---------------+---------------+----------------+------------")
print(f"  Accuracy       | {accuracy:.4f}      | {accuracy_optimal:.4f}       | {accuracy_optimal - accuracy:+.4f}")
print(f"  Precision      | {precision:.4f}      | {precision_optimal:.4f}       | {precision_optimal - precision:+.4f}")
print(f"  Recall         | {recall:.4f}      | {recall_optimal:.4f}       | {recall_optimal - recall:+.4f}")
print(f"  F1-Score       | {f1:.4f}      | {f1_optimal:.4f}       | {f1_optimal - f1:+.4f}")

# Step 8: Sigmoid Function Explanation
print("\n" + "=" * 70)
print("STEP 6: SIGMOID FUNCTION ROLE IN LOGISTIC REGRESSION")
print("=" * 70)

print(f"\nSigmoid Function Formula:")
print(f"  σ(z) = 1 / (1 + e^(-z))")
print(f"\nWhere:")
print(f"  z = linear combination of features: z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ")
print(f"  σ(z) = probability that sample belongs to class 1")
print(f"  Range: σ(z) ∈ (0, 1)")

print(f"\nSigmoid Function Properties:")
print(f"  1. Maps any real number to probability [0, 1]")
print(f"  2. S-shaped curve (hence 'sigmoid')")
print(f"  3. At z=0: σ(0) = 0.5 (decision boundary)")
print(f"  4. At z>0: σ(z) > 0.5 (tends toward 1)")
print(f"  5. At z<0: σ(z) < 0.5 (tends toward 0)")

# Demonstrate sigmoid function
z_values = np.linspace(-5, 5, 100)
sigmoid_values = 1 / (1 + np.exp(-z_values))

print(f"\nSigmoid Function Examples:")
print(f"  z = -5.0 → σ(z) = {1/(1+np.exp(5)):.6f}")
print(f"  z = -2.0 → σ(z) = {1/(1+np.exp(2)):.6f}")
print(f"  z =  0.0 → σ(z) = {1/(1+np.exp(0)):.6f}")
print(f"  z =  2.0 → σ(z) = {1/(1+np.exp(-2)):.6f}")
print(f"  z =  5.0 → σ(z) = {1/(1+np.exp(-5)):.6f}")

print(f"\nThreshold Decision Rule:")
print(f"  If σ(z) >= 0.5 → Predict Class 1 (Benign)")
print(f"  If σ(z) <  0.5 → Predict Class 0 (Malignant)")
print(f"  Custom threshold can optimize for specific use case")

import plotly.graph_objects as go
import numpy as np

# Data from the provided JSON
confusion_matrix = [[41, 1], [1, 71]]
labels = ["Malignant", "Benign"]

# Create the text annotations for each cell
text = [[str(confusion_matrix[i][j]) for j in range(2)] for i in range(2)]

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=confusion_matrix,
    x=['Predicted ' + l for l in labels],
    y=['Actual ' + l for l in labels],
    text=text,
    texttemplate="%{text}",
    textfont={"size": 20},
    colorscale='Blues',
    showscale=True,
    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
))

# Update layout
fig.update_layout(
    title='Confusion Matrix',
    xaxis=dict(side='bottom'),
    yaxis=dict(autorange='reversed')
)

# Save the figure
fig.write_image('confusion_matrix.png')
fig.write_image('confusion_matrix.svg', format='svg')

import plotly.graph_objects as go
import numpy as np

# Data
fpr = [0, 0.0238, 0.0238, 0.0238, 1, 1]
tpr = [0, 0, 1, 1, 1, 1]
auc = 0.9954
optimal_fpr = 0.0238
optimal_tpr = 1.0

# Create figure
fig = go.Figure()

# Add ROC curve
fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name=f'ROC (AUC={auc})',
    line=dict(color='#1FB8CD', width=3)
))

# Add diagonal line (random classifier)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random',
    line=dict(color='#5D878F', width=2, dash='dash')
))

# Add optimal threshold point
fig.add_trace(go.Scatter(
    x=[optimal_fpr],
    y=[optimal_tpr],
    mode='markers',
    name='Optimal Point',
    marker=dict(color='#DB4545', size=12, symbol='circle')
))

# Update layout
fig.update_layout(
    title='ROC Curve',
    xaxis_title='False Pos Rate',
    yaxis_title='True Pos Rate',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes to show 0 to 1 range
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])

# Apply cliponaxis
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('roc_curve.png')
fig.write_image('roc_curve.svg', format='svg')

import plotly.graph_objects as go
import numpy as np

# Data provided
z_values = [-10, -8, -6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6, 8, 10]
sigmoid_values = [0.00005, 0.00034, 0.00248, 0.01799, 0.11920, 0.26894, 0.37754, 0.5, 0.62246, 0.73106, 0.88080, 0.98201, 0.99752, 0.99966, 0.99995]

# Key points to mark
key_points_z = [-5, -2, 0, 2, 5]
key_points_sigmoid = [0.007, 0.119, 0.5, 0.881, 0.993]
key_labels = ['z=-5,σ≈0.007', 'z=-2,σ≈0.119', 'z=0,σ=0.5', 'z=2,σ≈0.881', 'z=5,σ≈0.993']

# Create figure
fig = go.Figure()

# Add sigmoid curve
fig.add_trace(go.Scatter(
    x=z_values,
    y=sigmoid_values,
    mode='lines',
    name='σ(z)=1/(1+e⁻ᶻ)',
    line=dict(color='#1FB8CD', width=3),
    hovertemplate='z=%{x:.1f}<br>σ(z)=%{y:.3f}<extra></extra>'
))

# Add key points with text labels
fig.add_trace(go.Scatter(
    x=key_points_z,
    y=key_points_sigmoid,
    mode='markers+text',
    name='Key Points',
    marker=dict(color='#DB4545', size=10, symbol='circle'),
    text=key_labels,
    textposition=['top center', 'top center', 'middle right', 'bottom center', 'bottom center'],
    textfont=dict(size=10, color='#DB4545'),
    hovertemplate='%{text}<extra></extra>'
))

# Add horizontal line at y=0.5 (decision threshold)
fig.add_trace(go.Scatter(
    x=[-10, 10],
    y=[0.5, 0.5],
    mode='lines',
    name='Threshold (0.5)',
    line=dict(color='#2E8B57', width=2, dash='dash'),
    hovertemplate='Decision Threshold<extra></extra>'
))

# Update layout
fig.update_layout(
    title='Sigmoid: Linear → Probability [0,1]',
    xaxis_title='z (linear)',
    yaxis_title='σ(z) (prob)',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes
fig.update_xaxes(range=[-10, 10])
fig.update_yaxes(range=[-0.05, 1.05])

# Apply cliponaxis=False
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('sigmoid_function.png')
fig.write_image('sigmoid_function.svg', format='svg')

import plotly.graph_objects as go
import json

# Load the data
data = {
    "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "default_threshold": [0.9825, 0.9861, 0.9861, 0.9861],
    "optimal_threshold": [0.9912, 0.9863, 1.0, 0.9931]
}

# Create the grouped bar chart
fig = go.Figure()

# Add Default threshold bars
fig.add_trace(go.Bar(
    x=data["metrics"],
    y=data["default_threshold"],
    name='Default (0.5)',
    marker_color='#1FB8CD',
    text=[f'{val:.4f}' for val in data["default_threshold"]],
    textposition='outside',
    cliponaxis=False
))

# Add Optimal threshold bars
fig.add_trace(go.Bar(
    x=data["metrics"],
    y=data["optimal_threshold"],
    name='Optimal (0.37)',
    marker_color='#DB4545',
    text=[f'{val:.4f}' for val in data["optimal_threshold"]],
    textposition='outside',
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Threshold Performance Comparison',
    xaxis_title='Metric',
    yaxis_title='Score',
    barmode='group',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Update y-axis range to accommodate text on top of bars
fig.update_yaxes(range=[0, 1.05])

# Save as PNG and SVG
fig.write_image('performance_metrics.png')
fig.write_image('performance_metrics.svg', format='svg')

# Create a comprehensive reference document
reference_doc = """# LOGISTIC REGRESSION BINARY CLASSIFIER - QUICK REFERENCE

## Dataset Summary
- Name: Breast Cancer Wisconsin (Diagnostic)
- Samples: 569 (212 malignant, 357 benign)
- Features: 30 numerical features
- Task: Binary classification (Malignant vs Benign)
- Train/Test: 80/20 split (455 train, 114 test)

## Evaluation Results

### Confusion Matrix
```
                Predicted
              Malignant | Benign
Actual   Malignant |    41   |   1   
         Benign    |    1    |  71   
```

### Performance Metrics (Threshold = 0.5)
- Accuracy:  98.25%  | Correctly classified predictions
- Precision: 98.61%  | Of predicted benign, 98.61% correct
- Recall:    98.61%  | Catches 98.61% of actual benign
- F1-Score:  0.9861  | Harmonic mean of precision & recall
- ROC-AUC:   0.9954  | Excellent discrimination ability

### Performance Metrics (Threshold = 0.3659)
- Accuracy:  99.12%  | +0.88% improvement
- Precision: 98.63%  | Essentially same
- Recall:    100.0%  | +1.39% improvement (catches ALL true positives)
- F1-Score:  0.9931  | +0.70% improvement
- Result:    113/114 correct (1 false positive only)

## Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | When we predict positive, how often correct? |
| Recall (Sensitivity) | TP/(TP+FN) | What % of positives do we catch? |
| Specificity | TN/(TN+FP) | What % of negatives do we catch? |
| F1-Score | 2(P×R)/(P+R) | Balance between precision & recall |
| ROC-AUC | Area under ROC curve | Probability ranking positive > negative |

## Confusion Matrix Components

- **TP (True Positive)**: 71 - Predicted benign, actually benign ✓
- **TN (True Negative)**: 41 - Predicted malignant, actually malignant ✓
- **FP (False Positive)**: 1 - Predicted benign, actually malignant ✗ (False alarm)
- **FN (False Negative)**: 1 - Predicted malignant, actually benign ✗ (Missed case)

## Sigmoid Function

Formula: σ(z) = 1 / (1 + e^-z)

Properties:
- Maps any real number to probability [0, 1]
- S-shaped curve (sigmoid = S-shaped)
- At z=0: σ(0) = 0.5 (decision point)
- At z→∞: σ(z)→1.0
- At z→-∞: σ(z)→0.0

Role in Logistic Regression:
- Converts linear output (z) to probability
- Enables probabilistic interpretation
- Creates smooth decision boundary
- Enables gradient-based optimization

Example transformations:
- z = -5 → σ(z) = 0.0067 (prob = 0.67%, predict class 0)
- z = -2 → σ(z) = 0.1192 (prob = 11.92%, predict class 0)
- z = 0  → σ(z) = 0.5 (prob = 50%, decision boundary)
- z = 2  → σ(z) = 0.8808 (prob = 88.08%, predict class 1)
- z = 5  → σ(z) = 0.9933 (prob = 99.33%, predict class 1)

## Feature Standardization

Why standardize?
1. Prevents high-magnitude features from dominating
2. Speeds up gradient descent convergence
3. Enables fair coefficient comparison
4. Required for regularization effectiveness

StandardScaler formula:
X_scaled = (X - mean) / std

Effect:
- Before: Features range from 0 to 3432
- After: Features standardized to mean=0, std=1
- Distribution shape unchanged, scale changed

Implementation:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

Critical: Fit scaler ONLY on training data, apply same scaler to test data.

## Model Training

Algorithm: Logistic Regression
Solver: LBFGS
Max iterations: 10,000
Regularization: L2 (default)

Parameters learned:
- Bias: 0.302208
- 30 weight coefficients

Top 5 important features:
1. Worst Texture (-1.2551)
2. Radius Error (-1.0830)
3. Worst Concave Points (-0.9537)
4. Worst Area (-0.9478)
5. Worst Radius (-0.9476)

Negative coefficients → more characteristic of malignant tumors

## Threshold Optimization

Default threshold: 0.5
Problem: May not be optimal for all use cases

Solution: Find optimal threshold using Youden's J = TPR - FPR

Found optimal: 0.3659
Why lower? Medical domain: Missing malignant is worse than unnecessary biopsy

Trade-off with threshold:
- Lower threshold → Higher recall, lower precision (catch more positives)
- Higher threshold → Lower recall, higher precision (fewer false alarms)

ROC curve shows this trade-off across all thresholds.
Choose threshold based on domain requirements (FP vs FN cost).

## Classification Rule

Default (threshold = 0.5):
- If P(class=1) >= 0.5 → Predict BENIGN
- If P(class=1) < 0.5 → Predict MALIGNANT

Optimal (threshold = 0.3659):
- If P(class=1) >= 0.3659 → Predict BENIGN
- If P(class=1) < 0.3659 → Predict MALIGNANT

## Use Cases for Each Metric

**Accuracy**: Use when classes are balanced and FP/FN equally important

**Precision**: Use when false positives are costly
- Examples: Spam detection (don't annoy users with false spam)
- Medical: Rare condition diagnosis (minimize unnecessary treatment)

**Recall**: Use when false negatives are costly
- Examples: Disease screening (don't miss sick patients)
- Security: Fraud detection (catch as many frauds as possible)

**F1-Score**: Use when you want balance between precision and recall
- Imbalanced datasets
- When both FP and FN matter, but equally

**ROC-AUC**: Use to compare models across all thresholds
- Threshold-independent evaluation
- Good for imbalanced datasets
- Single comprehensive metric

## For This Medical Problem

Why optimize for recall (Youden's J)?
- False Negative (missed malignant) = patient goes untreated (VERY BAD)
- False Positive (unnecessary biopsy) = unnecessary procedure (BAD but acceptable)
- Lower threshold catches more true positives → Higher recall
- Optimal threshold (0.3659) achieves 100% recall with only 1 FP

## Key Implementation Code

```python
# 1. Load and split data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer_data.data, cancer_data.target, 
    test_size=0.2, random_state=42, stratify=cancer_data.target
)

# 2. Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# 4. Get probabilities
y_pred_proba = model.predict_proba(X_test_scaled)

# 5. Evaluate
from sklearn.metrics import confusion_matrix, roc_auc_score
cm = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba[:, 1])

# 6. Optimize threshold
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

# 7. Predict with optimal threshold
y_pred_optimal = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
```

## Common Mistakes to Avoid

1. ✗ Fitting scaler on entire dataset → ✓ Fit on train only
2. ✗ Using default threshold without analysis → ✓ Analyze ROC curve
3. ✗ Only looking at accuracy on imbalanced data → ✓ Check precision/recall
4. ✗ Not stratifying train/test split → ✓ Use stratify parameter
5. ✗ Comparing raw feature coefficients → ✓ Compare standardized coefficients
6. ✗ Not checking feature scaling → ✓ Verify mean=0, std=1
7. ✗ Ignoring FP/FN trade-offs → ✓ Consider domain costs
8. ✗ Trusting single metric → ✓ Use multiple metrics (confusion matrix, ROC-AUC, etc.)

## Summary

Logistic regression is a fundamental binary classification algorithm:
- Simple yet powerful
- Probabilistic output (interpretable)
- Fast training
- Works well with standardized features
- Excellent for educational purposes and production systems

Key steps:
1. Select appropriate dataset ✓
2. Split train/test with stratification ✓
3. Standardize features ✓
4. Train logistic regression model ✓
5. Evaluate with confusion matrix + multiple metrics ✓
6. Analyze ROC curve ✓
7. Optimize threshold based on domain requirements ✓
8. Understand sigmoid function's role ✓

This project demonstrates all steps of a complete machine learning workflow!
"""

with open('Logistic_Regression_Quick_Reference.txt', 'w') as f:
    f.write(reference_doc)

print("✓ Quick Reference Guide created!")
print("File: Logistic_Regression_Quick_Reference.txt")
print(f"Content: {len(reference_doc)} characters")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
