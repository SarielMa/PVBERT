import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-i", "--Input", help = "the path to input model to be trained")

# Read arguments from command line
args = parser.parse_args()


base_path = "/home/lm2445/palmer_scratch/results_071325_class_no_pinfo"
if not os.path.exists(base_path):
    os.makedirs(base_path)
if args.Input == "eppc_bert_large":
    print("the inputing model is : % s" % args.Input)
    model_name = "/home/lm2445/project/bert-mlm-eppc-large/"
    output_dir = base_path + "/eppc_model_" + args.Input
elif args.Input == "eppc_bert_base":
    print("the inputing model is : % s" % args.Input)
    model_name = "/home/lm2445/project/bert-mlm-eppc_50epoch/"
    output_dir = base_path + "/eppc_model_" + args.Input
elif args.Input:
    print("the inputing model is : % s" % args.Input)
    model_name = args.Input
    output_dir = base_path + "/eppc_model_" + args.Input
else:
    raise Exception("Please choose a model")
   

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Processed Data
with open("stratified_train_data_no_pinfo.json", "r") as f:
    train_data = json.load(f)
with open("stratified_test_data_no_pinfo.json", "r") as f:
    val_data = json.load(f)
print (f"training size is {len(train_data)}")
print (f"test size is {len(val_data)}")

# Convert labels to unique indices
all_labels = set()
for dataset in [train_data, val_data]:
    for entry in dataset:
        all_labels.update(entry["labels"])

label_list = sorted(all_labels)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

class SentenceMultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry["text"]
        label_vector = [0] * len(self.label2id)
        for label in entry["labels"]:
            label_vector[self.label2id[label]] = 1  # multi-label one-hot

        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label_vector, dtype=torch.float)
        return encoding

# Create Dataset
train_dataset = SentenceMultiLabelDataset(train_data, tokenizer, label2id)
val_dataset = SentenceMultiLabelDataset(val_data, tokenizer, label2id)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
)




training_args = TrainingArguments(
    output_dir= output_dir,        # Where to save the model
    evaluation_strategy="epoch",      # Evaluate at the end of every epoch
    save_strategy="epoch",            # Save model only at the end of each epoch
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=10,               # Keep only the best model
    load_best_model_at_end=True,      # Load the best model at the end of training
    metric_for_best_model="eval_micro_f1", # Choose metric to determine best model
    greater_is_better=True,          # Lower eval_loss is better
    logging_dir="./logs",             # Log directory
    logging_strategy="steps",
    logging_steps=10,                 # Log every 10 steps
    #save_on_each_node=False,          # Save only once (not per node in multi-GPU)
    warmup_ratio=0.1,                 # 10% of the total steps warmup
    lr_scheduler_type="linear",       # linear decay
    # report_to="tensorboard",          # optional
)



def compute_metrics(pred):
    preds = (pred.predictions > 0).astype(int)
    labels = pred.label_ids
    return {
        "eval_micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "eval_macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "eval_precision": precision_score(labels, preds, average="micro", zero_division=0),
        "eval_recall": recall_score(labels, preds, average="micro", zero_division=0),
    }
# Train Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Load training state

# Find last checkpoint folder
checkpoint_dir = sorted([
    d for d in os.listdir(output_dir)
    if d.startswith("checkpoint-")
], key=lambda x: int(x.split("-")[1]))[-1]

with open(os.path.join(output_dir, checkpoint_dir, "trainer_state.json"), "r") as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]

# Extract metrics
steps = []
train_loss = []
eval_loss = []
f1_scores = []
eval_steps = []

for log in log_history:
    if "loss" in log:
        steps.append(log["step"])
        train_loss.append(log["loss"])
    if "eval_loss" in log:
        eval_steps.append(log["step"])
        eval_loss.append(log["eval_loss"])
        f1_scores.append(log.get("eval_micro_f1", None))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, train_loss, label="Train Loss", marker='o')
plt.plot(eval_steps, eval_loss, label="Eval Loss", marker='x')

if any(f1_scores):
    plt.plot(eval_steps, f1_scores, label="Eval Micro F1", marker='s')

plt.xlabel("Training Steps")
plt.ylabel("Metric Value")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(output_dir, "training_plot.png"), dpi=300)
plt.close()
