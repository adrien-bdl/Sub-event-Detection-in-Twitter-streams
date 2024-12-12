"""
From original df of tweets, produces df of BERT embeddings for a subsample of N_tweets per minute
"""
import pandas as pd
import os
import torch
from transformers import AutoModel, AutoTokenizer
from src.TweetNormalizer import normalizeTweet
import sys

def get_BERT_embedding(tweets, model, tokenizer, batch_size=64):
    """Computes BERT embeddings from a list of tweets"""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N_batches = len(tweets) // batch_size
    t_outputs = []
    model.eval()
    print(f"--- Embedding ---")
    with torch.no_grad():
        for batch in range(N_batches):
            print(f"\t {batch * batch_size:8d} / {len(tweets)} ({int(batch * batch_size / len(tweets) * 100):3d} %)", end="\r")

            inputs = tokenizer(tweets[batch * batch_size: (batch + 1) * batch_size], return_tensors="pt", padding=True,
                                truncation=True, max_length=128) ##
            outputs = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            token_type_ids=inputs["token_type_ids"].to(device))
            t_outputs.append(outputs.pooler_output.cpu())

        if len(t_outputs) * batch_size < len(tweets):
            inputs = tokenizer(tweets[N_batches * batch_size:], return_tensors="pt", padding=True,
                               truncation=True, max_length=128) ##
            outputs = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            token_type_ids=inputs["token_type_ids"].to(device))
            t_outputs.append(outputs.pooler_output.cpu())
        t_outputs = torch.cat(t_outputs)
    
    return t_outputs

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        mode="eval"
    else:
        mode = sys.argv[1]

    if mode == "train":
        path_data = "challenge_data/train_tweets"
        path_write = "challenge_data/train_BERT"
    
    else:
        path_data = "challenge_data/train_tweets"
        path_write = "challenge_data/train_BERT"

    print(mode)
    exit()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_files = os.listdir(path_data)

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

    N_tweets = 400 # number of tweets per minute to embedd
    preambule = "Does the following tweet refer to a recent and significant event in the soccer match ? [SEP] " # None if no preambule

    for i in range(len(data_files)):
        print(f"--- File {i + 1} / {len(data_files)} ---")

        filename = data_files[i]
        df = pd.read_csv(f"{path_data}/" + filename)

        ### Subsampling
        sampled_df = df.groupby("ID").apply(
            lambda x: x.sample(n=N_tweets, replace=True if len(x) < N_tweets else False).sort_values(by="Timestamp")
            ).reset_index(drop=True)    
        
        ### Preprocessing
        sampled_df["Tweet"] = sampled_df["Tweet"].apply(normalizeTweet)
        if preambule:
            sampled_df["Tweet"] = sampled_df["Tweet"].apply(lambda x: preambule + x)
            
        ### Embedding
        tweet_tensors = get_BERT_embedding(list(sampled_df["Tweet"]), bertweet, tokenizer, batch_size=128)
        tweet_df = pd.DataFrame(tweet_tensors)
        df_embeddings = pd.concat([sampled_df, tweet_df], axis=1).drop(columns="Tweet")
        df_embeddings.to_pickle(f"{path_write}/df_{filename[:-4]}.pkl")