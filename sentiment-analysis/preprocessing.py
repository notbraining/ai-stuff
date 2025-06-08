!huggingface-cli login
!pip install --upgrade datasets huggingface_hub fsspec
!pip install transformers accelerate evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from datasets import DatasetDict
import numpy as np
import evaluate


ds = load_dataset("KayEe/flipkart_sentiment_analysis")
# you can take a subset of ds to decrease training time
subset_size_train = 500
subset_size_test = 500
train_subset = ds['train'].select(range(subset_size_train))
test_subset = ds['test'].select(range(subset_size_test))


#replace ds with subset ds
subset_ds = DatasetDict({
    'train': train_subset,
    'test': test_subset
})
ds=subset_ds

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3) # 0=pos, 1=neg, 2=neutral
def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True)

tokenized_ds = ds.map(tokenize_function, batched=True)

# Rename columns to fit the model's expected format
tokenized_ds = tokenized_ds.rename_column("output", "label")

