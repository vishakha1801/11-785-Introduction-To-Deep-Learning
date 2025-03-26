import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.A = A
        batch_size, in_channels, input_width = A.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1
        # TODO Create a new array Z with the correct shape

        Z = np.zeros((batch_size, in_channels, output_width)) # TODO

        # TODO Fill in the values of Z by upsampling A

        Z[:, :, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # TODO Slice dLdZ by the upsampling factor to get dLdA
        dLdA = np.zeros_like(self.A)
        dLdA = dLdZ[:,:,::self.upsampling_factor]  # TODO

        return dLdA



class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # TODO Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)

        self.A = A
        input_width = A.shape[2]
        output_width = (input_width - 1) // self.downsampling_factor + 1

        Z = A[:, :, ::self.downsampling_factor]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        input_width = self.A.shape[2]
        # TODO Create a new array dLdA with the correct shape

        dLdA = np.zeros((batch_size, in_channels, input_width))  # TODO

        # TODO Fill in the values of dLdA with values of A as needed
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = self.upsampling_factor * (input_height - 1) + 1
        output_width = self.upsampling_factor * (input_width - 1) + 1

        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        Z[:,:,::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # TODO Slice dLdZ by the upsampling factor to get dLdA

        dLdA = np.zeros_like(self.A)
        dLdA[:, :, :, :] = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        # TODO Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)
        self.A = A

        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor] # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape

        # TODO Create a new array dLdA with the correct shape
        dLdA = np.zeros_like(self.A)

        # TODO Fill in the values of dLdA with values of A as needed
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA
