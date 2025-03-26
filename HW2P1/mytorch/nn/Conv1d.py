# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A  # Store for backward pass

        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))  # TODO

        for i in range(output_size):
            input_slice = A[:, :, i:i + self.kernel_size]
            Z[:, :, i] = np.tensordot(input_slice, self.W, axes=([1, 2], [1, 2]))

        Z += self.b[None, :, None]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        _, in_channels, kernel_size = self.W.shape
        input_size = output_size + kernel_size - 1

        dLdA = np.zeros((batch_size, in_channels, input_size))
        self.dLdW.fill(0)
        self.dLdb.fill(0)

        for i in range(output_size):
            dLdA[:, :, i:i + kernel_size] += np.tensordot(dLdZ[:, :, i], self.W, axes=([1], [0]))

            self.dLdW += np.tensordot(dLdZ[:, :, i], self.A[:, :, i:i + kernel_size], axes=([0], [0]))

        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # Sum over batch and width

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=0)

        stride_A = self.conv1d_stride1.forward(A_padded)

        # Call Conv1d_stride1
        Z_conv = self.conv1d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample1d.forward(stride_A)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ_upsampled = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv1d_stride1.backward(dLdZ_upsampled)  # TODO

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded

        return dLdA
