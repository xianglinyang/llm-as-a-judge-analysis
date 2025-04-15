import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving the model and vectorizer
import os # Import os for directory creation
import numpy as np
import pandas as pd

# --- Configuration ---
# --- Data Loading Configuration ---
DATA_DIR = "/data2/xianglin/data/preference_leakage/output" # Base directory for data
MODEL_DIR = "/data2/xianglin/data/preference_leakage"
DATA_FILES = [
    f"{DATA_DIR}/gemini_data.npy",
    f"{DATA_DIR}/gpt4o_data.npy",
    f"{DATA_DIR}/llama_data.npy"
]

# --- Classifier Configuration ---
TEST_SIZE = 0.2                 # Proportion of data for testing
CLASSIFIER_TYPE = 'LogisticRegression' # Options: 'NaiveBayes', 'LogisticRegression', 'SVM'
SEED = 42                       # For reproducible train/test split
MODEL_SAVE_PATH = f"{MODEL_DIR}/extracted_feature_bow_classifier_{CLASSIFIER_TYPE}.joblib"

# --- Ensure output directory exists ---
# Extract directory from save paths if needed
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- 1. Load Data ---
print("Loading feature data...")    # load data into a dataframe
# selected_feature_idxs = np.random.choice(range(67), size=60, replace=False)

df = pd.DataFrame()
for i, data_file in enumerate(DATA_FILES):
    print(f"Loading file: {data_file} with label {i}")
    # feature_vecs = np.load(data_file)[:, selected_feature_idxs]
    feature_vecs = np.load(data_file)
    feature_vecs = feature_vecs.tolist()
    labels = [i]*len(feature_vecs)
    # current df
    current_df = pd.DataFrame(feature_vecs, columns=[f"feature_{j}" for j in range(len(feature_vecs[0]))])
    current_df['label'] = labels
    df = pd.concat([df, current_df], ignore_index=True)
    print(f"Loaded {len(df)} items from {len(DATA_FILES)} files")

print(f"Total loaded {len(df)} items")
# # Optional: Shuffle the DataFrame rows (good practice before splitting)
# df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# --- 2. Split Data ---
# 67 features
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED, # for reproducibility
    stratify=y      # ensure similar label distribution in train/test
)
print(f"\nData split: Train={len(X_train)}, Test={len(X_test)}")


# --- 3. Classifier Training ---
print(f"\nTraining Classifier: {CLASSIFIER_TYPE}...")

classifier_params = {
    'random_state': SEED,
    'penalty': 'l1',    # l1 for sparsity
    'C': 1.0,  # regularization strength
    'fit_intercept': False,
    # Add other common parameters if needed across classifiers
}

if CLASSIFIER_TYPE == 'NaiveBayes':
    classifier = MultinomialNB() # Doesn't take random_state typically
    # Adjust alpha (smoothing) if needed: classifier = MultinomialNB(alpha=0.1)
elif CLASSIFIER_TYPE == 'LogisticRegression':
    classifier = LogisticRegression(max_iter=1000, solver='liblinear', **classifier_params)
    # Consider regularization tuning: C=1.0 (default), penalty='l2'
elif CLASSIFIER_TYPE == 'SVM':
    # LinearSVC parameters can be sensitive
    classifier = LinearSVC(max_iter=2000, dual='auto', **classifier_params)
    # Consider tuning C parameter (regularization strength)
else:
    raise ValueError("Unsupported CLASSIFIER_TYPE")

classifier.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluation ---
print("\nEvaluating classifier on test data...")
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# Get class labels for report (assuming labels are 0, 1, 2...)
target_names = [f"Model_{i}" for i in sorted(y.unique())]
report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
# Use integer labels directly if target_names cause issues with heatmap
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix ({CLASSIFIER_TYPE})')
plt.tight_layout() # Adjust layout
plt.show()

# --- 6. Save Model and Vectorizer ---
print(f"\nSaving model to {MODEL_SAVE_PATH}")
joblib.dump(classifier, MODEL_SAVE_PATH)
print("Model and vectorizer saved.")

# --- Optional: Load and Predict on new data ---
# print("\nLoading model for prediction example...")
# loaded_classifier = joblib.load(MODEL_SAVE_PATH)
# loaded_vectorizer = joblib.load(VECTORIZER_SAVE_PATH)
# new_texts = ["This sounds like Model A.", "A different kind of text from B."]
# new_texts_vec = loaded_vectorizer.transform(new_texts)
# predictions = loaded_classifier.predict(new_texts_ vec)
# print(f"Predictions for new texts: {predictions}")
# print(f"Predicted labels map to: {[target_names[p] for p in predictions]}")