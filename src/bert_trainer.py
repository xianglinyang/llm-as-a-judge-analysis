# from preference_leakage paper
# Then, we finetune a BERT-base-uncased model in the training set. The model is trained for 3 epochs with a learning rate of
# 2e-5, a batch size of 16 for both training and evaluation, and a weight decay of 0.01, with evaluations conducted at the end
# of each epoch.

import torch
import json
import random
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, ClassLabel

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd # Optional: if loading from pandas DataFrame


DATA_DIR = "/data2/xianglin/data/preference_leakage"

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"  # Or "roberta-base", "distilbert-base-uncased", etc.
DATA_FILES = [
    f"{DATA_DIR}/UltraFeedback_sampled_30000_gemini_LlamaFactory.json",
    f"{DATA_DIR}/UltraFeedback_sampled_30000_gpt4_LlamaFactory.json",
    f"{DATA_DIR}/UltraFeedback_sampled_30000_llama_LlamaFactory.json"
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

# Make sure the order of files corresponds to labels 0, 1, 2...
EXPECTED_JSON_OUTPUT_KEY = "output" # Key in JSON containing the text
SEED = 42 # For reproducibility

TEXT_COLUMN = "text"             # Name of the column containing the text
LABEL_COLUMN = "label"            # Name of the column containing the integer labels
NUM_LABELS = 3                    # !! IMPORTANT: Set this to the number of distinct models !!
TEST_SIZE = 0.2                   # Proportion of data to use for validation/testing
OUTPUT_DIR = f"{DATA_DIR}/bert_classifier_results" # Directory to save model and results
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# --- 1. Load Data ---
print("Loading data...")
data_list = [] # Use a list of dictionaries to build DataFrame

with open(QUESTION_TYPE_DATA_FILE, "r", encoding="utf-8") as f:
    question_type_data = json.load(f)
# process question type data
question_type_mapping = {}
for item in question_type_data:
    if 'question' in item.keys() and 'categorization' in item.keys() and 'question category' in item['categorization'].keys():
        question_type_mapping[item['question']] = item['categorization']['question category']



for i, data_file in enumerate(DATA_FILES):
    label = i # Assign label based on file index (0, 1, 2...)
    print(f"Loading file: {data_file} with label {label}")
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # merge
        count = 0
        for i in range(len(data)):
            instruction = data[i]['instruction']
            qt = question_type_mapping.get(instruction, None)
            data[i]['question_type'] = qt
            if qt is not None:
                count += 1
        print(f"Total {count} question types found for {data_file}")

        count = 0
        for item in data:
            # Check if the expected key exists and its value is not None/empty string
            if isinstance(item, dict) and EXPECTED_JSON_OUTPUT_KEY in item and item[EXPECTED_JSON_OUTPUT_KEY]:
                 text_output = item[EXPECTED_JSON_OUTPUT_KEY]
                 if isinstance(text_output, str) and text_output.strip() and item['question_type'] is not None:
                    data_list.append({TEXT_COLUMN: text_output, LABEL_COLUMN: label, 'question_type': item['question_type']})
                    count += 1
        print(f"  Loaded {count} valid items.")
    except FileNotFoundError:
        print(f"  Error: File not found - {data_file}")
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON from - {data_file}")
    except Exception as e:
        print(f"  An unexpected error occurred loading {data_file}: {e}")

# Convert the list of dictionaries into a Pandas DataFrame
if not data_list:
    raise ValueError("No valid data loaded. Please check DATA_FILES paths and JSON structure.")

df = pd.DataFrame(data_list)
print(f"\nTotal valid items loaded into DataFrame: {len(df)}")
print("DataFrame Info:")
df.info()
print("\nLabel distribution:")
print(df[LABEL_COLUMN].value_counts())

# Optional: Shuffle the DataFrame rows (good practice before splitting)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# --- 2. Split Data ---
X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]
question_types = df['question_type']

X_train, X_test, y_train, y_test, question_type_train, question_type_test = train_test_split(
    X, y, question_types,
    test_size=TEST_SIZE,
    random_state=SEED, # for reproducibility
    stratify=y      # ensure similar label distribution in train/test
)
print(f"\nData split: Train={len(X_train)}, Test={len(X_test)}")

# Create proper DataFrames for train and test sets
train_df = pd.DataFrame({
    TEXT_COLUMN: X_train,
    LABEL_COLUMN: y_train,
})

test_df = pd.DataFrame({
    TEXT_COLUMN: X_test,
    LABEL_COLUMN: y_test,
})

dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df)
})

print("Dataset Splits:")
print(dataset_dict)

# --- 2. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # padding=True/max_length will be handled by DataCollator
    return tokenizer(examples[TEXT_COLUMN], truncation=True, max_length=512, padding="max_length")

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# Remove original text column to avoid issues with data collator
tokenized_datasets = tokenized_datasets.remove_columns([TEXT_COLUMN])
# Rename label column if necessary to 'labels' (expected by Trainer)
if LABEL_COLUMN != 'label':
     tokenized_datasets = tokenized_datasets.rename_column(LABEL_COLUMN, "label")
# Set format for PyTorch
tokenized_datasets.set_format("torch")


# --- 3. Model Loading ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# --- 4. Training Setup ---

# Data Collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics Calculation Function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted') # Use 'weighted' for imbalance
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",       # Save checkpoint at the end of each epoch
    load_best_model_at_end=True, # Load the best model based on validation loss
    metric_for_best_model="eval_f1", # Use F1 score to determine the best model
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,            # Log training loss every 100 steps
    fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
    report_to="none",     # Log to tensorboard (install if needed: pip install tensorboard)
    # push_to_hub=False,         # Set to True to push model to Hugging Face Hub
)

# --- 5. Trainer Initialization ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 6. Training ---
print("Starting training...")
trainer.train()

# --- 7. Evaluation ---
print("\nEvaluating final model on validation set...")
eval_results = trainer.evaluate()
print("\nEvaluation Results:")
print(eval_results)

eval_per_question_type_dict = {}

# --- 8. Save Final Model & Tokenizer ---
print("\nSaving best model...")
trainer.save_model(f"{OUTPUT_DIR}/best_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")

print("\nTraining and evaluation complete.")
print(f"Model saved to: {OUTPUT_DIR}/best_model")

# --- 9 Evaluation on each question type ---
for question_type in QUESTION_TYPE_NOUNS:
    print(f"\nEvaluating on {question_type}...")
    select_x = X_test[question_type_test == question_type]
    select_y = y_test[question_type_test == question_type]
    
    q_df = pd.DataFrame({
        TEXT_COLUMN: select_x,
        LABEL_COLUMN: select_y
    })

    # Create dataset from pandas and tokenize it properly
    q_dataset = Dataset.from_pandas(q_df.copy())
    # Tokenize this dataset in the same way as the main datasets
    q_tokenized = q_dataset.map(tokenize_function, batched=True)
    # Remove original text column and set format to pytorch
    q_tokenized = q_tokenized.remove_columns([TEXT_COLUMN])
    q_tokenized.set_format("torch")
    
    eval_per_question_type_dict[question_type] = q_tokenized

for question_type in QUESTION_TYPE_NOUNS:
    print(f"\nEvaluating on {question_type}...")
    # Check if we have any data for this question type
    if len(eval_per_question_type_dict[question_type]) == 0:
        print(f"No data available for {question_type}")
        continue

    eval_results_per_q = trainer.evaluate(eval_per_question_type_dict[question_type])
    print(eval_results_per_q)


# # --- Optional: Prediction on a new text ---
# from transformers import pipeline
# pipe = pipeline("text-classification", model=f"{OUTPUT_DIR}/best_model", tokenizer=f"{OUTPUT_DIR}/best_model")
# new_text = "This text might be from the second model."
# result = pipe(new_text)
# print(f"\nPrediction for '{new_text}': {result}")
