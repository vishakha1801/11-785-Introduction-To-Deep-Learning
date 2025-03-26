import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug
        self.W = np.zeros((out_features, in_features))  # TODO
        self.b = np.zeros((out_features, 1)) # TODO
        self.dLdW = None
        self.dLdb = None


    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.A = A  # TODO
        self.N = A.shape[0]  # TODO - store the batch size parameter of the input A

        # Think how can `self.ones` help in the calculations and uncomment below code snippet.
        self.ones = np.ones((self.N, 1))

        Z = np.dot(A, self.W.T) + np.dot(self.ones, self.b.T)

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        dLdA = np.dot(dLdZ, self.W)  # TODO
        self.dLdW = np.dot(dLdZ.T, self.A)  # TODO
        self.dLdb = np.dot(dLdZ.T, self.ones).reshape(-1, 1) # TODO

        if self.debug:
            self.dLdA = dLdA

        return dLdA
