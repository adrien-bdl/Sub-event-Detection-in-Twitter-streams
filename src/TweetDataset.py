"""
Definition of the dataset class for training and pretraining of the CNN binary classifier
"""

import numpy as np
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    """X has shape (N_minutes, N_tweets, N_features)"""

    def __init__(self, X, y, sequence_length):
        self.X = X.float()
        self.y = y.unsqueeze(1).float()
        self.N_tweets = self.X.shape[1]
        self.sequence_length = sequence_length
        self.replace = False if self.sequence_length <= self.N_tweets else True

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ids_sample = np.sort(np.random.choice(a=self.N_tweets, size=self.sequence_length, replace=self.replace))
        return self.X[idx, ids_sample, :].permute(1, 0), self.y[idx]
    
class NextTweetDataset(Dataset):
    """X has shape (N_minutes, N_tweets, N_features)"""

    def __init__(self, X, sequence_length, K):
        """K is the number of tweets averaged to predict"""
        self.X = X.reshape((X.shape[0] * X.shape[1], -1)).float()
        self.sequence_length = sequence_length
        self.K = K

    def __len__(self):
        return len(self.X) - 2 * self.sequence_length - 2 * self.K
    
    def __getitem__(self, idx):
        idxs_1 = np.sort(np.random.choice(a=2*self.sequence_length, replace=False,
                         size=self.sequence_length))
        idxs_2 = np.sort(np.random.choice(a=2*self.K, replace=False,
                         size=self.K))

        return (self.X[idx:idx + 2 * self.sequence_length][idxs_1].permute(1, 0),
                self.X[idx + 2 * self.sequence_length: idx + 2 * self.sequence_length + 2 * self.K][idxs_2].mean(axis=0))

    

