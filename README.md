# AI/ML Engineer Coding Assessment

--

## EDA
- In this dataset the data provided is a text data.
- The NLP Task is summarisation
- There are three subdatasets in the dataset folder
    - The dataset is downloaded from [Google Drive](https://drive.google.com/drive/u/1/folders/1q9Hd3ChNzamrHpWP_MlT-LYi6EYDjDKc)
    - The Sub-datasets are 
        - In-Abs (Trainset: 7030 Testset: 100)
        - In-Ext (Trainset: 50 Testset: Not Provided)
        - UK-Abs (Trainset: 693 Testset: 100)

    - In the folder there are `judgements` and their summaries in `summary`

- Dataset token size for input text and summary text is as follows: (code in `src/token_size.ipynb`)


|                                   | Input Text | Output Text (Summary) |
|-----------------------------------|------------|-----------------------|
| Maximum context length in dataset | 182,927     | 38,134                 |
| Minimum context length in dataset | 267        | 0                     |
| Average context length in dataset | 5800.73    | 1149.13               |

>NOTE: To time constraint other datasets are not evaluated/trained (`IN-Ext` and `UK-Abs`) but the code is general, just need to change the path in CLI to do the analysis


## Model
- Selected `t5-small` model as it is small and loadable into the resourse we have right now. Also trains faster.
- Trainsformers are pre-trained and these models are good at the specialised task after finetuning
- This model has 60M Parameters
- The model is a full transformer (encoder+decoder)(Vaswani et al architecture) which makes it good at encoding the input text and decode the text to the required human readable format (summary)
- Other models like BERT or GPT are just for encoding or decoding (auto generative) respectively. Which makes the model not good for the summarization task
- Other encoder + decoder models are mBART, T5 which are too large considering the resource at disposal
> NOTE: Finetuned model is not uploaded as it is very large

# Metric used

## Usage
### Installation
- Create python (Tested on Python 3.8.16; Ubuntu 20.04 LTS) environment
- Install the requrirements using `pip install -r requirements.txt`
- Download the dataset as mentioned in Dataset section

### Dataset
- Download the dataset from the repository [Google Drive](https://drive.google.com/drive/u/1/folders/1q9Hd3ChNzamrHpWP_MlT-LYi6EYDjDKc)
- Paste it inside the directory `data` after creating the directory
- Extract the folders

### Training
- In the `--train-data` argument give the folder directory in which the `judement` and `summary` folders are present. Similarly for the `--test-data`
- We can use the following code
```
python src/train.py --train-data ./data/dataset/IN-Abs/train-data --test-data ./data/dataset/IN-Abs/test-data
```
- There are some additional parameters that could be passed 
    - **--model** Huggingface or local model to be trained is to be provided here (default t5-small)
    - **--epoch** that takes in number of epochs the model should be run (default is 4)
    - **--batch** Selecting the batch size of the dataset (default is 32)
    - **--lr** The learning rate of the model (default is 2e-5)
    - **--cuda** To select the CUDA Device (default is 2)
    - **--max-length** for context length (default is 512)

> NOTE: Trainined model will be saved in the directory `./data/training`

> NOTE: The local model must be a director containing the config.json, pytorch_model.bin, special_tokens_map.json, tokenizer.json, tokenizer_config.json, and vocab.txt 

#### Results on the system are
For ROUGE Metric
- Rouge1: 0.0169 (Unigram overlap)
- Rouge2: 0.0086 (Bigram)
- RougeL: 0.015 (Longest Common Subsequence)

Values are low but could be improved by (either of)

    - cleaning the dataset
    - Better annotating the summary 
    - hyperparameter tunning on the dataset
    - use larger model like T5 or mT5(if multilingual) 
    - Increase context length

### Generating

The `src/generate.py` function provides functionality to generate the summary of the text given as input. To use use the code 
```
python src/generate.py --input-file ./data/dataset/IN-Ext/judgement/1953_L_1.txt  --trained-model ./data/training/checkpoint-880
```

> NOTE: You can specify the trained model or the huggingface repo of the model in the `--trained-model` parameter. Also the input text file is accepted by `--input-file` file. 

> NOTE: The code generates the output to the STDOUT but cn redirect to a file using pipes. Example append `| tee output.txt` at the end of the command being executed in the CLI.

> NOTE: The local model must be a director containing the config.json, pytorch_model.bin, special_tokens_map.json, tokenizer.json, tokenizer_config.json, and vocab.txt

> NOTE: Finetuned model is not uploaded as it is very large

### Testing
To Run test cases on the model and get the accuracy of the model

Just use the `src/trainer.py` same as (Trainer) but with extra parameter `--only-evaluate` with value `True`. By default the value is `False`
