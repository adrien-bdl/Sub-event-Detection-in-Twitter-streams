"""
Utils for data manipulation
"""

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def read_BERT(path_dfs="../challenge_data/train_BERT_400", N_files="all"):
    """Reads N_files dfs of BERT embeddings in a directory in pkl format"""

    li = []
    for i, filename in enumerate(os.listdir(path_dfs)):
        df_ = pd.read_pickle(f"{path_dfs}/" + filename)
        li.append(df_)
        if N_files != "all" and i + 1 >= N_files:
            break

    df = pd.concat(li, ignore_index=True)
    
    return df


def MATCH_ID_train_test_split(df, test_size=0.25):
    """Splits the df without mixing the MATCH_IDs. A proportion p_test of the matches
       is used for testing"""

    id_train, id_test = train_test_split(df["MatchID"].unique(), test_size=test_size)

    return df[df["MatchID"].isin(id_train)], df[df["MatchID"].isin(id_test)]


def df_to_tensors(df, eval=False):
    """Takes a df of embeddings and returns a tensor of dimension (n_periods, n_tweets, dim_embedding)"""

    X, y = [], []
    ids = []
    for id, df_id in df.groupby("ID"):
        if eval:
            X.append(torch.tensor(df_id.drop(columns=["ID", "MatchID", "PeriodID"]).values))
            ids.append(id)
        else:
            y.append(df_id.iloc[0]["EventType"])
            X.append(torch.tensor(df_id.drop(columns=["ID", "MatchID", "PeriodID", "EventType"]).values))

    X = torch.stack(X)
    y = torch.LongTensor(y) #

    if eval:
        return X, ids
    
    return X, y


