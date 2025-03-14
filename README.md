# Kaggle INF554

## Adrien BINDEL, Victor DESHORS, Rodrigue REIBEL

This repository contains the implementation of a machine learning pipeline for real-time detection of sub-events during football matches based on tweets. The project leverages state-of-the-art natural language processing techniques to identify and classify sub-events such as goals, penalties, and red cards from large-scale social media data.

## 1. Structure of the code

- challenge_data/
    - train_tweets/ and eval_tweets/ contain csv files of the original data
    - BERT_embeddings/eval_BERT/ and BERT_embeddings/train_BERT/ folders will contain BERT embeddings once scripts/create_BERT_df.py is run

- player_data/
    - contain the list of the players who played during the 2010 and 2014 World Cups and that is used to anonymized the data

- src/
    - contains class and function definitions for data manipulation, training, tuning and evaluation.

- scripts/
    - create_BERT_df.py : creates the df of BERT embeddings from a given df of tweets
    - cluster_classifier_tuning.py : parameter tuning for the cluster classifier model
    - cluster_classifier_eval.py : evaluation on eval dataset for the cluster classifier model
    - CNN_classifier_training.py : training of the CNN model
    - CNN_classifier_eval.py : evaluation on eval dataset for the CNN classifier model
    - cluster_classifier_GloVe.py : evaluation on eval dataset with GloVe as embedding model and LogisticRegression for the classification

- research/
    - contains a notebook used for parameter-tuning and model's selection.

- cv_results/
    - stores results of parameter search

- models/
    - stores trained DL models (of type CNNBinaryClassifier)

- submissions/
    - stores submission files

## 2. Code usage

<!-- 1. *Original data* :
    - Put the original data in folders challenge_data/train_tweets/ and challenge_data/eval_tweets/ -->

1. *First create BERT embeddings for train and eval dataset* :
    - run ``python -m scripts.create_BERT_df train``
    - run ``python -m scripts.create_BERT_df eval``

2. *Evaluate the cluster-classifier model*:

    - run ``python -m scripts.cluster_classifier_eval``

3. *Evaluate the cluster-classifier with GloVe for embedding and LogisticRegression as the classifier model*:

    - run ``python -m scripts.cluster_classifier_GloVe``

4. *Evaluate the CNN classifier model*:

- First train the CNN binary classifier model
    - ``run python -m scripts.CNN_classifier_training``
- Then eval the CNN binary classifier model
    - ``run python -m scripts.CNN_classifier_eval``
