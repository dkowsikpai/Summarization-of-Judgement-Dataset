import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import pipeline
from torch.utils.data import Dataset
import os
import evaluate

"""
This code uses huggingface library with transformers to summarise the task
"""

################################################################
# Parsing command line arguments
parser = argparse.ArgumentParser(description ='Code to train the model')
parser.add_argument('--input-file', type = str, help ='train data file path')
parser.add_argument('--trained-model', type = str, help ='trained model path huggingface/local')
parser.add_argument('--cuda', type = str, help ='select the cuda/gpu', default="0")

args = parser.parse_args()

#Setting seeds for reproducability 
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


# Configurations
model_checkpoint = "t5-small" # Huggingface model repo name

max_length = 512 # Context Length

CUDA = args.cuda
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
device = device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
##################################################################

# Loading model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

with open(args.input_file, "r") as f:
    l = f.readlines()
    l = [x.strip() for x in l]
    text = ''.join(l)

# Using inference pipeline to genereate the summary of the given text
summarizer = pipeline("summarization", model=args.trained_model)
summary_text = list(summarizer(text))
print(summary_text[0]["summary_text"])


