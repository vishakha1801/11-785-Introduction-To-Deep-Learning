# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')



class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]  # TODO: Add the layers in the correct order

    def init_weights(self, weights):
        w1, w2, w3 = weights

        w1_transposed = w1.T
        self.conv1.conv1d_stride1.W = w1_transposed.reshape(8, 8, 24).transpose(0, 2, 1)  # (8, 24, 8)

        w2_transposed = w2.T
        self.conv2.conv1d_stride1.W = w2_transposed.reshape(16, 1, 8).transpose(0, 2, 1)  # (16, 8, 1)

        w3_transposed = w3.T
        self.conv3.conv1d_stride1.W = w3_transposed.reshape(4, 1, 16).transpose(0, 2, 1)  # (4, 16, 1)

    def forward(self, A):
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ  # Initialize
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        return self.forward(A)

    def init_weights(self, weights):
        w1, w2, w3 = weights

        w1_transposed = w1[:48, :2].T
        self.conv1.conv1d_stride1.W = w1_transposed.reshape(2, 2, 24).transpose(0, 2, 1)  # (2, 24, 2)

        w2_transposed = w2[:4, :8].T
        self.conv2.conv1d_stride1.W = w2_transposed.reshape(8, 2, 2).transpose(0, 2, 1)  # (8, 2, 2)

        w3_transposed = w3.T
        self.conv3.conv1d_stride1.W = w3_transposed.reshape(4, 2, 8).transpose(0, 2, 1)  # (4, 8, 2)

    def forward(self, A):
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ  # Initialize gradient
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA