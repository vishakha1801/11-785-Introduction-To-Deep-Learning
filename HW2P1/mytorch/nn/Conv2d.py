import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        batch_size, in_channels, input_height, input_width = A.shape
        out_channels, in_channels, kernel_size, kernel_size = self.W.shape
        output_height = input_height - kernel_size + 1
        output_width = input_width - kernel_size + 1


        Z = np.zeros((batch_size, out_channels, output_height, output_width))  # TODO

        for i in range(output_height):
            for j in range(output_width):
                Z[:, :, i, j] = np.tensordot(
                    A[:, :, i:i + kernel_size, j:j + kernel_size],
                    self.W, axes=([1, 2, 3], [1, 2, 3])
                )

        Z += self.b[None, :, None, None]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, input_height, input_width = self.A.shape
        out_channels, in_channels, kernel_size, kernel_size = self.W.shape
        output_height, output_width = dLdZ.shape[2], dLdZ.shape[3]

        self.dLdW = np.zeros_like(self.W) # TODO
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))  # TODO
        dLdA = np.zeros_like(self.A)  # TODO

        for i in range(output_height):
            for j in range(output_width):
                self.dLdW += np.tensordot(
                    dLdZ[:, :, i, j],
                    self.A[:, :, i:i + kernel_size, j:j + kernel_size],
                    axes=([0], [0])
                )
        flipped_W = np.flip(self.W, axis=(2, 3))

        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (kernel_size - 1, kernel_size - 1), (kernel_size - 1, kernel_size - 1)), mode='constant')

        for i in range(input_height):
            for j in range(input_width):
                dLdA[:, :, i, j] = np.tensordot(
                    padded_dLdZ[:, :, i:i + kernel_size, j:j + kernel_size], flipped_W, axes=([1, 2, 3], [0, 2, 3])
                )


        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=0)

        # Call Conv2d_stride1
        Z_conv = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z_conv)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_upsampled)  # TODO

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded

        return dLdA
