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
import json
import pandas as pd

# --- Configuration ---
# --- Data Loading Configuration ---
DATA_DIR = "/data2/xianglin/data/preference_leakage/output" # Base directory for data
MODEL_DIR = "/data2/xianglin/data/preference_leakage/extracted_feature_classifier"
DATA_FILES = [
    f"{DATA_DIR}/gemini.json",
    f"{DATA_DIR}/gpt4.json",
    f"{DATA_DIR}/llama.json"
]

QUESTION_TYPE_DATA_FILE = '/home/xianglin/git_space/llm-as-a-judge-analysis/analysis/dataset_categorization/ultrafeedback_categorization.json'
QUESTION_TYPE_NOUNS = [
    "Computer Science & Programming",
    "Mathematics & Statistics",
    "Science & Engineering",
    "Business & Finance",
    "Writing & Communication",
    "Social & Daily Life",
    "Others"
]

EXPECTED_JSON_OUTPUT_KEY = 'feature_vec'
FEATURE_VEC_COLUMN = 'feature_vec'
QUESTION_COLUMN = 'question'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
VECTOR_SIZE = 67

# --- Classifier Configuration ---
TEST_SIZE = 0.2                 # Proportion of data for testing
CLASSIFIER_TYPE = 'SVM' # Options: 'NaiveBayes', 'LogisticRegression', 'SVM'
SEED = 42                       # For reproducible train/test split
MODEL_SAVE_PATH = f"{MODEL_DIR}/{CLASSIFIER_TYPE}.joblib"

# --- Ensure output directory exists ---
# Extract directory from save paths if needed
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# --- 1. Load Data ---
print("Loading data...")

with open(QUESTION_TYPE_DATA_FILE, "r", encoding="utf-8") as f:
    question_type_data = json.load(f)
# process question type data
question_type_mapping = {}
for item in question_type_data:
    if 'question' in item.keys() and 'categorization' in item.keys() and 'question category' in item['categorization'].keys():
        question_type_mapping[item['question']] = item['categorization']['question category']

df = pd.DataFrame()
for i, data_file in enumerate(DATA_FILES):
    label = i # Assign label based on file index (0, 1, 2...)
    print(f"Loading file: {data_file} with label {label}")
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # merge
        count = 0
        for j in range(len(data)):  # Changed i to j to avoid overwriting label
            instruction = data[j]['instruction']
            qt = question_type_mapping.get(instruction, None)
            data[j]['question_type'] = qt
            if qt is not None:
                count += 1
        print(f"Total {count} question types found for {data_file}")

        count = 0
        data_list = []
        question_type_list = []
        for item in data:
            if isinstance(item, dict) and EXPECTED_JSON_OUTPUT_KEY in item and item[EXPECTED_JSON_OUTPUT_KEY]:
                feature_vec = item[EXPECTED_JSON_OUTPUT_KEY]
                question_type = item['question_type']
                text = item['instruction']
                if isinstance(feature_vec, list) and len(feature_vec) == VECTOR_SIZE and question_type is not None and text is not None:
                    question_type_list.append(question_type)
                    data_list.append(feature_vec)
                    count += 1
        current_df = pd.DataFrame(data_list, columns=[f"feature_{j}" for j in range(len(data_list[0]))])
        current_df[QUESTION_COLUMN] = question_type_list
        current_df[LABEL_COLUMN] = [label]*len(data_list)
        df = pd.concat([df, current_df], ignore_index=True)
        print(f"Loaded {len(df)} items from {len(DATA_FILES)} files")

    except FileNotFoundError:
        print(f"  Error: File not found - {data_file}")
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON from - {data_file}")
    except Exception as e:
        print(f"  An unexpected error occurred loading {data_file}: {e}")

# Optional: Shuffle the DataFrame rows (good practice before splitting)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# --- 2. Split Data ---
# 67 features
feature_columns = [f"feature_{j}" for j in range(VECTOR_SIZE)]
X = df[feature_columns].astype(float)
y = df[LABEL_COLUMN].astype(int)
question_type = df[QUESTION_COLUMN].astype(str)

print(f"Label distribution: {y.value_counts()}")
print(f"Collect {len(y.unique())} labels: {sorted(y.unique())}")

X_train, X_test, y_train, y_test, question_type_train, question_type_test = train_test_split(
    X, y, question_type,
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

# # Plot Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=target_names, yticklabels=target_names)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title(f'Confusion Matrix ({CLASSIFIER_TYPE})')
# plt.tight_layout() # Adjust layout
# plt.show()


# --- 5.1 Compute test accuracy for each question type ---
# Test data: X_test, y_test, question_type_test

# compute accuracy for each question type
for question_type in QUESTION_TYPE_NOUNS:
    select_x = X_test[question_type_test == question_type]
    select_y = y_test[question_type_test == question_type]
    select_y_pred = y_pred[question_type_test == question_type]
    accuracy = accuracy_score(select_y, select_y_pred)
    print(f"Accuracy for {question_type}: {accuracy:.4f}")

    # Get class labels for report (assuming labels are 0, 1, 2...)
    target_names = [f"Model_{i}" for i in sorted(select_y.unique())]
    report = classification_report(select_y, select_y_pred, target_names=target_names, zero_division=0)
    conf_matrix = confusion_matrix(select_y, select_y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

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