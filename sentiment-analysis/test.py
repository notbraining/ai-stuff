#give it a fixed seed for consistency 
train_dataset = tokenized_ds["train"].shuffle(seed=42)
eval_dataset = tokenized_ds["test"].shuffle(seed=42)

# metric evaluates accuracy
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    #weight_decay gives a penalty for higher weights, this is supposed to penalize overfitting (memborizing)
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

#training
trainer.train()

#testing
eval_results = trainer.evaluate()
eval_results
