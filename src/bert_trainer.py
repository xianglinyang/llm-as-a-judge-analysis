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
TEXT_COLUMN = "text"             # Name of the column containing the text
LABEL_COLUMN = "label"            # Name of the column containing the integer labels
NUM_LABELS = 3                    # !! IMPORTANT: Set this to the number of distinct models !!
TEST_SIZE = 0.1                   # Proportion of data to use for validation/testing
OUTPUT_DIR = f"{DATA_DIR}/bert_classifier_results" # Directory to save model and results
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# --- 1. Load and Prepare Data ---
dataset_dict = {
    TEXT_COLUMN: list(),
    LABEL_COLUMN: list()
}
for i, data_file in enumerate(DATA_FILES):
    with open(data_file, "r") as f:
        data = json.load(f)
    for item in data:
        if item["output"] is None:
            continue
        dataset_dict[TEXT_COLUMN].append(item["output"])
        dataset_dict[LABEL_COLUMN].append(i)

# shuffle the data
dataset = Dataset.from_dict(dataset_dict)
# --- Split Data ---
# If you only have one split, create train/test splits
# Do I need to shuffle the data?
train_test_split_data = dataset.train_test_split(test_size=TEST_SIZE, shuffle=True)

dataset_dict = DatasetDict({
    'train': train_test_split_data['train'],
    'validation': train_test_split_data['test'] # Use test split as validation for Trainer
    # 'test': test_valid_split_data['test'] # Keep a final test set separate if desired
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
    eval_dataset=tokenized_datasets["validation"],
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

# --- 8. Save Final Model & Tokenizer ---
print("\nSaving best model...")
trainer.save_model(f"{OUTPUT_DIR}/best_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")

print("\nTraining and evaluation complete.")
print(f"Model saved to: {OUTPUT_DIR}/best_model")

# # --- Optional: Prediction on a new text ---
# from transformers import pipeline
# pipe = pipeline("text-classification", model=f"{OUTPUT_DIR}/best_model", tokenizer=f"{OUTPUT_DIR}/best_model")
# new_text = "This text might be from the second model."
# result = pipe(new_text)
# print(f"\nPrediction for '{new_text}': {result}")
