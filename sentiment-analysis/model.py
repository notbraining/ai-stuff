!huggingface-cli login
!pip install --upgrade datasets huggingface_hub fsspec
!pip install transformers accelerate evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
# you can take a subset of ds to decrease training tim
subset_size_train = 500
subset_size_test = 500
train_subset = ds['train'].select(range(subset_size_train))
test_subset = ds['test'].select(range(subset_size_test))
ds = load_dataset("KayEe/flipkart_sentiment_analysis")

from datasets import DatasetDict
#replace ds with subset ds
subset_ds = DatasetDict({
    'train': train_subset,
    'test': test_subset
})
ds=subset_ds
