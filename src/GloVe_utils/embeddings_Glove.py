import pandas as pd

import numpy as np
from sklearn.cluster import KMeans

from tqdm import tqdm

### Functions to get the embeddings


def get_avg_embedding(tweet, model, vector_size=200):
    """compute the average word vector for a tweet"""
    tweet = str(tweet)
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if (
        not word_vectors
    ):  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


def get_main_kernel_embedding(df, n_clusters=3):
    if "EventType" in df.columns:
        vector_columns = ["EventType"] + [str(i) for i in range(200)]
    else:
        vector_columns = [str(i) for i in range(200)]

    df[vector_columns] = (
        df[vector_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df[vector_columns])
    clusters_df = pd.DataFrame(kmeans.labels_, columns=["cluster"], index=df.index)
    df = pd.concat([df, clusters_df], axis=1)
    main_kernel_label = df["cluster"].value_counts().idxmax()
    df["is_main_kernel"] = (df["cluster"] == main_kernel_label).astype(int)
    main_kernel_vectors = df[df["cluster"] == main_kernel_label][vector_columns]
    avg_main_kernel_embedding = main_kernel_vectors.mean(axis=0)

    return avg_main_kernel_embedding, df


def process_tweet_file(filename, embeddings_model, vector_size, eval=False):
    """
    Process a single tweet file, get embeddings and compute kernel embeddings.
    """
    if eval:
        df = pd.read_csv(
            f"../../challenge_data/cleaned_tweets/cleaned_eval_tweets/{filename}"
        )
    else:
        df = pd.read_csv(
            f"../../challenge_data/cleaned_tweets/cleaned_train_tweets/{filename}"
        )

    tweet_vectors = np.vstack(
        [
            get_avg_embedding(tweet, embeddings_model, vector_size)
            for tweet in tqdm(df["Tweet"], desc="Processing Tweets")
        ]
    )
    tweet_df = pd.DataFrame(tweet_vectors)
    tweet_df.columns = tweet_df.columns.astype(str)

    df = df.drop(columns=["Timestamp"])

    period_features = pd.concat([df, tweet_df], axis=1)

    # Group the tweets by MatchID, PeriodID, and ID
    grouped = period_features.groupby(["MatchID", "PeriodID", "ID"])

    results = []
    for (match_id, period_id, tweet_id), group in tqdm(
        grouped, desc="Processing the kernels"
    ):
        avg_embedding, _ = get_main_kernel_embedding(group)
        results.append((match_id, period_id, tweet_id, *avg_embedding))

    columns = ["MatchID", "PeriodID", "ID", "EventType"] + [str(i) for i in range(200)]

    main_kernel_embeddings = pd.DataFrame(results, columns=columns)

    return main_kernel_embeddings
