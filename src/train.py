import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import os
import evaluate

"""
This code uses huggingface library with transformers to summarise the task
"""

################################################################
# Parsing command line arguments
parser = argparse.ArgumentParser(description ='Code to train the model')
parser.add_argument('--train-data', type = str, help ='train data file path')
parser.add_argument('--test-data', type = str, help ='test data file path')
parser.add_argument('--epoch', type = int, help ='number of epochs', default=4)
parser.add_argument('--model', type = str, help ='Model to train', default="t5-small")
parser.add_argument('--batch', type = int, help ='batch size', default=32)
parser.add_argument('--lr', type = float, help ='learning rate', default=2e-5)
parser.add_argument('--cuda', type = str, help ='select the cuda/gpu', default="0")
parser.add_argument('--max-length', type = int, help ='Context Length', default=512)
parser.add_argument('--only-evaluate', type = bool, help ='Do not train and only evaluate', default=False)

args = parser.parse_args()

#Setting seeds for reproducability 
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


# Configurations
model_checkpoint = args.model # Huggingface model repo name
BATCH_SIZE = args.batch
EPOCH = args.epoch # Number of epochs
LR = args.lr # Learning rate. Just using the one generally used. Needs hyperparameter tuning to improve
MAX_LENGTH = args.max_length

CUDA = args.cuda
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
device = device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
##################################################################

# Reading the training and testing files
train_files = {
    "text": os.listdir(args.train_data+'/judgement'),
    "summary": os.listdir(args.train_data+'/summary')
}
test_files = {
    "text": os.listdir(args.test_data+'/judgement'),
    "summary": os.listdir(args.test_data+'/summary')
}

print("Number of train set:", len(train_files["text"]))
print("Number of test set", len(test_files["text"]))


# Loading model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

# Setting Dataset transformer
class SummaryDataset(Dataset):
    def __init__(self, dataset_files, tokenizer, max_length=None, n_samples=None, dstype="train"):
        """
            TODO: Documentation
        """
        self.texts = []
        self.labels = []
        self.max_length = tokenizer.model_max_length if max_length is None else max_length 

        base_dir = args.train_data if dstype == "train" else args.test_data

        # Reading 
        count = 0
        for file in dataset_files["text"]:
            # Reading and tokenizing the text file 
            with open(base_dir+'/judgement/'+file, 'r') as f:
                l = f.readlines()
                l = [x.strip() for x in l]
                text = ''.join(l)
                # Tokenizer creates the token for the document and returns the pytorch tensor with padding for parallism in GPU
                tok = tokenizer(text, return_tensors="pt",  truncation=True, max_length=max_length, padding="max_length")
                self.texts.append(tok)

            # Reading and tokenizing the summary file corresponding to the text file
            with open(base_dir+'/summary/'+file, 'r') as f:
                l = f.readlines()
                l = [x.strip() for x in l]
                text = ''.join(l)
                # Tokenizer creates the token for the document and returns the pytorch tensor with padding for parallism in GPU
                tok = tokenizer(text, return_tensors="pt",  truncation=True, max_length=max_length, padding="max_length")
                self.labels.append(tok)

            count += 1
            if n_samples is not None and count == n_samples:
                break
  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids':self.texts[idx]["input_ids"][0], "attention_mask": self.texts[idx]["attention_mask"][0], "labels":self.labels[idx]["input_ids"][0]}


# Loading and tokenizing the dataset
train_dataset = SummaryDataset(train_files, tokenizer, max_length=MAX_LENGTH, dstype="train")
test_dataset = SummaryDataset(test_files, tokenizer, max_length=MAX_LENGTH, dstype="test")


# Evaluation metric
rouge = evaluate.load("rouge") # A metric in which the value is between 0 and 1; higher the better

# Compute metric function- executed after every epoch
def compute_metrics(eval_pred):
    predictions, labels = eval_pred # Contains both predicted and true labels (labels)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) # Int Token to String
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) # Removing padding
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) # Int Token to String

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True) # Computing the metric

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    num_train_epochs=EPOCH,
    output_dir="data/training",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3, # Number of models that need to be saved
    predict_with_generate=True,
    warmup_steps=500,
    fp16=True,
    # push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics,
)


if not args.only_evaluate:
    # Train the model
    trainer.train()
else: 
    trainer.evaluate()

