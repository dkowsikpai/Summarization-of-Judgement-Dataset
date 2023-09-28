# AI/ML Engineer Coding Assessment

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

## Model
- Selected t5-small model as it is small and loadable into the resourse we have right now. Also trains faster.
- Trainsformers are pre-trained and these models are good at the specialised task after finetuning
- This model has 60M Parameters

# Metric used

## Useage
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
        - `--epoch` that takes in number of epochs the model should be run (default is 4)
        - `--batch` Selecting the batch size of the dataset (default is 32)
        - `--lr` The learning rate of the model (default is 2e-5)
        - `--cuda` To select the CUDA Device (default is 2)

> NOTE: Trainined model will be saved in the directory `./data/training`

### Testing



