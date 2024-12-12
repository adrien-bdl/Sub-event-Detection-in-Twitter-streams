#%% Imports
import os
from src import CNNBinaryClassifier
import pandas as pd
import torch
from src import utils, DL_utils
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.TweetDataset import TweetDataset, NextTweetDataset
from src.DL_utils import compute_accuracy
import pickle

pd.options.display.float_format = '{:.2f}'.format  # Limit to 2 decimal places
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    # %% Read the data
    path_dfs = "challenge_data/train_BERT"
    df_BERT = utils.read_BERT(path_dfs=path_dfs)

    df_train, df_test = utils.MATCH_ID_train_test_split(df_BERT.drop(columns=["Timestamp", "Tweet"]))

    X_train, y_train = utils.df_to_tensors(df_train)
    X_test, y_test = utils.df_to_tensors(df_test)

    # %% Define model

    # We tune these parameters and assess the performance of the model
    model_params = {
        "N_channels": 768, 
        "sequence_length": 128, 
        "conv_kernel_size": 5, 
        "pool_kernel_size": 2, 
        "hidden_channels": [128, 64],
        "fc_hidden_layer": 64,
        "stride": 2
    }

    model = CNNBinaryClassifier.CNNBinaryClassifier(**model_params).to(device)
    print(f"Tensor size after convolutions : {model._calculate_flatten_size()}")

    ########## PRETRAINING ##########
    # the model trains to predict the mean embedding of the K next tweets, given the 
    # embeddings of sequence_length consecutive tweets
    # %% Pretraining parameters

    K = 10
    p_hidden_layer = 1024
    p_batch_size = 64
    p_n_epochs = 1
    p_lr = 1e-3
    p_loss_fn = torch.nn.SmoothL1Loss()
    p_optimizer = torch.optim.Adam(model.parameters(), lr=p_lr)
    
    p_train_dataset = NextTweetDataset(X_train, sequence_length=model.sequence_length, K=K)
    p_train_loader = DataLoader(p_train_dataset, batch_size=p_batch_size, shuffle=True)

    # %% Pretraining

    model.set_pretraining(hidden_layer=p_hidden_layer)
    model.to(device)

    DL_utils.train(model, p_optimizer, p_loss_fn, p_train_loader, n_epochs=p_n_epochs,
                n_prints=20)

    ########## TRAINING ##########
    # The model trains to classify each period, given a sample of sequence_length
    # tweets from that period
    # %% Training parameters

    batch_size = 16
    n_epochs = 10
    n_evals = 10 # number of validation epochs during training
    lr = 1e-4
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)


    freq_pos = y_train.sum() / len(y_train)
    weights = [1 / freq_pos if y_train[i] == 1 else 1 / (1 - freq_pos) for i in range(len(y_train))]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(y_train))

    train_dataset = TweetDataset(X_train, y_train, sequence_length=model.sequence_length)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

    test_dataset = TweetDataset(X_test, y_test, sequence_length=model.sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #%% Train the model

    model.set_training()
    model.to(device)

    DL_utils.train(model, optimizer, loss_fn, train_loader, test_loader, n_epochs=n_epochs,
                n_prints=5, n_evals=n_evals)

    #%% Evaluation

    for s in [0.4, .45, 0.5, 0.55, 0.6]: # vary the threshold for classification
        print(f"s = {s}")
        eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        accuracy, y_pred = compute_accuracy(model, eval_loader, s=s, n_epochs=10)
        print(f"Test accuracy:  {accuracy:.3f}")

    # %% Save the model
    save_path = "models/CNN_classifier_128.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, save_path)


