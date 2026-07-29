# Date Fruit Classification using Feed-Forward Neural Network (FNN)
A PyTorch-based Feed-Forward Neural Network that classifies date fruits into 7 varieties using shape, color, and texture features extracted from images. The project also compares model performance with and without PCA-based dimensionality reduction.

# 📌 Overview
Task: Multi-class classification (7 date fruit varieties)
Model: Feed-Forward Neural Network (ANN) built with PyTorch
Dataset: Date Fruit Dataset — 898 samples, 34 numeric features
Highlight: Trained two versions of the model — one on raw scaled features, one on PCA-reduced features — and compared their convergence and accuracy
# 🍇 Dataset
Property	 Value
Samples    898
Features	 34 (morphological, shape, and color/texture statistics)
Classes	   7 — BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY
File	     DateFruit_Dataset.csv

# Feature groups include:
Shape/geometry: AREA, PERIMETER, MAJOR_AXIS, MINOR_AXIS, ECCENTRICITY, ROUNDNESS, SOLIDITY, etc.
Shape factors: SHAPEFACTOR_1 – SHAPEFACTOR_4
Color/texture statistics (RGB channels): Mean, StdDev, Skewness, Kurtosis, Entropy per channel
Wavelet features: ALLdaub4RR, ALLdaub4RG, ALLdaub4RB
# ⚙️ Methodology
## Preprocessing
Split features (X) and target (y = Class)
Label-encoded the 7 class names
Train/test split: 80/20
Standardized features using StandardScaler
Converted to PyTorch tensors and wrapped in DataLoader (batch size = 32)
## Model Architecture (FNN)
   Input Layer  → Linear(n_features, 64) → ReLU
   Hidden Layer → Linear(64, 64)          → ReLU
   Output Layer → Linear(64, n_classes)
Loss: CrossEntropyLoss
Optimizer: Adam (lr = 0.001)
Epochs: 100
## Dimensionality Reduction Experiment
Applied PCA and analyzed cumulative explained variance
Retrained a smaller FNN (n_components → 32 → 32 → n_components) on the top 8 principal components
Compared training loss curves and test accuracy against the original model
# 📊 Results
Model	Final Training Loss	Test Accuracy
FNN (without PCA)	≈ 0.03	95.56%
FNN (with PCA, 8 components)	≈ 0.15	68.33%
## Conclusion: 
The PCA-based model converges with a lower initial loss but plateaus higher and generalizes worse. The model trained on the full feature set achieves both lower final training loss and significantly higher test accuracy, making it the better choice for this dataset.

# 🛠️ Tech Stack
Python
PyTorch
scikit-learn (StandardScaler, LabelEncoder, train_test_split, PCA, metrics)
Pandas / NumPy
Matplotlib / Seaborn
# 📈 Future Improvements
Try alternate PCA component counts to find the accuracy/dimensionality sweet spot
Add dropout / batch normalization to reduce overfitting
Experiment with deeper architectures or learning-rate scheduling
Add cross-validation for more robust accuracy estimates
