"""
We run the CNN binary classifier model on the evaluation dataset.
"""
# %% Imports
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from src.TweetDataset import TweetDataset
from src.DL_utils import CNN_kaggle_eval
from src import utils
from src import CNNBinaryClassifier
import pickle

pd.options.display.float_format = '{:.2f}'.format  # Limit to 2 decimal places
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    # %% Read the evaluation data
    path_dfs = "challenge_data/eval_BERT"
    df_BERT = utils.read_BERT(path_dfs=path_dfs).drop(columns="Timestamp")

    # %% Read training data
    path_dfs_train = "challenge_data/train_BERT"

    df_train_ = utils.read_BERT(path_dfs=path_dfs_train).drop(columns="Timestamp")
    df_train = df_train_.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    X_train = df_train.drop(columns=['EventType', 'MatchID', 'ID', 'PeriodID']).values
    y_train = df_train['EventType'].values

    # %% Load model

    path_model = "models/CNN_classifier_128.pkl"

    with open(path_model, "rb") as f:
        model = pickle.load(f)

    model.to(device)

    # %% Preparing data

    eval_dataset = TweetDataset(df_BERT, sequence_length=model.sequence_length, eval=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, drop_last=False)

    #%% Evaluation

    y_pred, ids = CNN_kaggle_eval(model, eval_loader, n_epochs=1)
    df_results = pd.DataFrame(data={
        "ID": ids,
        "EventType": y_pred
    })
    df_results.to_csv(f"challenge_data/submissions/{path_model[7:-4]}.csv", index=False)
