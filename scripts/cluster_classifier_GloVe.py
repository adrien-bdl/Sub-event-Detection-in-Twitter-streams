from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import os

# from xgboost import XGBClassifier
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

import gensim.downloader as api
import nltk
import os

from src.GloVe_utils.preprocessing import TweetCleaner, replace_names_in_tweet
from src.GloVe_utils.embeddings_Glove import process_tweet_file
from src.GloVe_utils.creation_new_features import add_tweet_features


if __name__ == "__main__":

    #####################
    ### PREPROCESSING ###
    #####################

    cleaner = TweetCleaner()

    train_data_path = "../challenge_data/train_tweets"

    for i, filename in tqdm(enumerate(os.listdir(train_data_path))):
        df = pd.read_csv(f"{train_data_path}/" + filename)
        df["Tweet"] = df["Tweet"].apply(lambda x: cleaner.clean(x))
        df["Tweet"] = df["Tweet"].progress_apply(
            lambda x: replace_names_in_tweet(x, filename)
        )
        df.to_csv(
            "../cleaned_glove_challenge_data/cleaned_train_tweets/cleaned_" + filename,
            index=False,
        )
        print(f"{filename} preprocessed")

    #################
    ### EMBEDDING ###
    #################

    # Download some NLP models for processing, optional
    nltk.download("stopwords")
    nltk.download("wordnet")
    # Load GloVe model with Gensim's API
    embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings

    vector_size = 200  # 200-dimensional GloVe embeddings

    li = []
    for filename in os.listdir(
        "../../challenge_data/cleaned_tweets/cleaned_train_tweets"
    ):
        print(f"Processing {filename}")
        main_kernel_embeddings = process_tweet_file(
            filename, embeddings_model, vector_size
        )
        li.append(main_kernel_embeddings)

    imported_df = pd.concat(li, ignore_index=True)

    ####################
    ### NEW FEATURES ###
    ####################

    path_data = "../challenge_data/train_tweets"
    files = []

    for i, filename in enumerate(os.listdir(path_data)):
        if filename == ".ipynb_checkpoints":
            continue
        df = pd.read_csv(f"{path_data}/" + filename)
        files.append(df)

    df = pd.concat(files, ignore_index=True)

    df_new_features = add_tweet_features(df)

    ##########################################################
    ### JOIN TRAINING SETs WITH EMBEDDINGS AND NEW FEATURES ###
    ##########################################################

    imported_df["ID"] = imported_df["ID"].astype(str)
    df_new_features["ID"] = df_new_features["ID"].astype(str)

    columns_to_keep = [
        "nb_tweets_per_minute",
        "nb_RT_per_min",
        "nb_@_per_min",
        "Match_time",
        "nb_consecutive_letters_per_minute",
        "nb_smileys_per_minute",
        "Exclamation_Count_per_minute",
        "Question_Count_per_minute",
    ]
    period_features = pd.concat(
        [imported_df.set_index("ID"), df_new_features.set_index("ID")[columns_to_keep]],
        axis=1,
    ).reset_index()

    ###########################################################
    ### SELECTION OF THE FEATURES TO KEEP IN PANDAS DF df_X ###
    ###########################################################

    liste_features_to_drop = ["MatchID", "ID", "Match_time"]

    df_X = period_features.drop(columns=liste_features_to_drop + ["EventType"])
    df_y = period_features["EventType"]

    ###########
    ### PCA ###
    ###########

    # Sélection des colonnes pour la PCA
    columns_to_pca = [str(i) for i in range(1, 200)]  # Colonnes '1' à '199'
    X_pca_input = df_X[columns_to_pca]

    # number of Principal Components
    N = 50

    pca = PCA(n_components=N)  # Réduction à N features
    X_pca = pca.fit_transform(X_pca_input)
    pca_columns = [f"PCA_{i+1}" for i in range(N)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df_X.index)
    columns_to_keep = [col for col in df_X.columns if col not in columns_to_pca]
    df_X = pd.concat([df_X[columns_to_keep], df_pca], axis=1)

    ################
    ### TRAINING ###
    ################

    # BEST CLASSIFIER MODEL : LogisticRegression
    # Best parameters : {'C': 100, 'max_iter': 5000, 'penalty': 'l2', 'solver': 'lbfgs'}
    # Best accuracy (cross-validation) : 0.6261189783098777

    # CROSS VALIDATION SUR TOUTE LA DATA + TOUTES LES FEATURES
    X = df_X.values
    y = df_y.values

    clf = LogisticRegression(
        random_state=42, max_iter=5_000, C=100, penalty="l2", solver="lbfgs"
    )

    cv_scores = cross_val_score(
        clf, X, y, cv=5, scoring="accuracy"
    )  # 5-fold cross-validation

    print("Accuracy moyenne (cross-validation) :", np.mean(cv_scores))
    print("Accuracy std (cross-validation) :", np.std(cv_scores))

    #############################
    ###-----------------------###
    ### FOR KAGGLE SUBMISSION ###
    ###-----------------------###
    #############################

    #####################
    ### PREPROCESSING ###
    #####################

    eval_data_path = "../challenge_data/eval_tweets"
    for i, filename in enumerate(os.listdir(eval_data_path)):
        df = pd.read_csv(f"{eval_data_path}/" + filename)
        df["Tweet"] = df["Tweet"].apply(lambda x: cleaner.clean(x))
        df["Tweet"] = df["Tweet"].progress_apply(
            lambda x: replace_names_in_tweet(x, filename)
        )
        df.to_csv(
            "../cleaned_glove_challenge_data/cleaned_eval_tweets/cleaned_" + filename,
            index=False,
        )
        print(f"{filename} done")

    #################
    ### EMBEDDING ###
    #################

    vector_size = 200  # 200-dimensional GloVe embeddings

    li = []
    for filename in os.listdir(
        "../../challenge_data/cleaned_tweets/cleaned_eval_tweets"
    ):
        print(f"Processing {filename}")
        main_kernel_embeddings = process_tweet_file(
            filename, embeddings_model, vector_size, eval=True
        )
        li.append(main_kernel_embeddings)

    imported_df = pd.concat(li, ignore_index=True)

    ####################
    ### NEW FEATURES ###
    ####################

    path_data = "../challenge_data/eval_tweets"
    files = []

    for i, filename in enumerate(os.listdir(path_data)):
        if filename == ".ipynb_checkpoints":
            continue
        df = pd.read_csv(f"{path_data}/" + filename)
        files.append(df)

    df = pd.concat(files, ignore_index=True)

    df_new_features = add_tweet_features(df)

    ##########################################################
    ### JOIN TRAINING SETs WITH EMBEDDINGS AND NEW FEATURES ###
    ##########################################################

    imported_df["ID"] = imported_df["ID"].astype(str)
    df_new_features["ID"] = df_new_features["ID"].astype(str)

    columns_to_keep = [
        "nb_tweets_per_minute",
        "nb_RT_per_min",
        "nb_@_per_min",
        "Match_time",
        "nb_consecutive_letters_per_minute",
        "nb_smileys_per_minute",
        "Exclamation_Count_per_minute",
        "Question_Count_per_minute",
    ]
    period_features = pd.concat(
        [imported_df.set_index("ID"), df_new_features.set_index("ID")[columns_to_keep]],
        axis=1,
    ).reset_index()

    ###########################################################
    ### SELECTION OF THE FEATURES TO KEEP IN PANDAS DF df_X ###
    ###########################################################

    liste_features_to_drop = ["MatchID", "ID", "Match_time"]

    df_X_eval = period_features.drop(columns=liste_features_to_drop)

    ###########
    ### PCA ###
    ###########

    # Sélection des colonnes pour la PCA
    columns_to_pca = [str(i) for i in range(1, 200)]  # Colonnes '1' à '199'
    X_pca_input = df_X_eval[columns_to_pca]

    # number of Principal Components
    N = 50

    pca = PCA(n_components=N)  # Réduction à N features
    X_pca = pca.fit_transform(X_pca_input)
    pca_columns = [f"PCA_{i+1}" for i in range(N)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df_X_eval.index)
    columns_to_keep = [col for col in df_X_eval.columns if col not in columns_to_pca]
    df_X_eval = pd.concat([df_X[columns_to_keep], df_pca], axis=1)

    ################################
    ### TRAINING ON FULL DATASET ###
    ################################

    X = df_X.values
    y = df_y.values

    clf = LogisticRegression(
        random_state=42, max_iter=5_000, C=100, penalty="l2", solver="lbfgs"
    ).fit(X, y)

    ##################
    ### PREDICTION ###
    ##################

    preds = clf.predict(df_X_eval.values)

    pred_df = pd.DataFrame()
    pred_df["EventType"] = preds
    pred_df["ID"] = df_X_eval["ID"]

    pred_df.to_csv("logistic_predictions.csv", index=False)
