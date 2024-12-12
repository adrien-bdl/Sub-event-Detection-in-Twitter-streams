"""
Definition of the 1D CNN binary classifier. This model takes an input of shape N_channels x sequence_length,
corresponding to a sequence of sequence_length N_channels-dimensional tweet embeddings. It performs a binary
classification, and can be pretrained on a next tweet prediction-type task.  
"""
from torch import nn
import torch.nn.functional as F
import numpy as np

class CNNBinaryClassifier(nn.Module):

    def __init__(self, N_channels=768, sequence_length=128, conv_kernel_size=5, pool_kernel_size=2, 
                 hidden_channels=[256, 128, 64], fc_hidden_layer=64, stride=1):
        """
        Initialize the adaptive CNN binary classifier.
        """
        super(CNNBinaryClassifier, self).__init__()
        
        self.N_channels = N_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.hidden_channels = hidden_channels
        self.stride = stride
        self.sequence_length = sequence_length
        self.fc_hidden_layer = fc_hidden_layer
        self.training_fc = None
        self.pretraining_fc = None
        
        # Dynamically create convolutional blocks based on hidden_channels
        conv_blocks = []

        in_channels = N_channels
        for out_channels in hidden_channels:
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size,
                              stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                    nn.MaxPool1d(kernel_size=pool_kernel_size)  # Halve the sequence length
                )
            )
            in_channels = out_channels  # Update in_channels for the next layer
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        
        self.set_training()
        

    def set_pretraining(self, hidden_layer, restart_pretraining=False):
        """Sets the CNN in pretraining mode (task : next tweet prediction)"""

        if self.pretraining_fc is None or restart_pretraining:
            self.pretraining_fc = self.fc = nn.Sequential(
                nn.Linear(self._calculate_flatten_size(), hidden_layer),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_layer, self.N_channels),
                nn.Tanh()
            )
        self.fc = self.pretraining_fc

    def set_training(self, restart_training=False):
        """Sets the CNN in training mode (task : binary classification)"""

        if self.training_fc is None or restart_training:
            self.training_fc = nn.Sequential(
                nn.Linear(self._calculate_flatten_size(), self.fc_hidden_layer),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.fc_hidden_layer, 1),
                nn.Sigmoid()
            )
        self.fc = self.training_fc

    def _calculate_flatten_size(self):
        """Calculate the size of the flattened output after the convolutional blocks."""

        sequence_length = self.sequence_length
        for _ in self.hidden_channels:
            sequence_length = np.floor((sequence_length - self.conv_kernel_size) / self.stride + 1) # convolution
            sequence_length = np.floor((sequence_length - self.pool_kernel_size) / self.pool_kernel_size + 1) # pooling
        
        return int(sequence_length * self.hidden_channels[-1])

    def forward(self, x):
        """
        Forward pass of the model, for next tweet prediction or binary classification depending
        on self.fc
        """        
        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        return x