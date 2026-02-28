import numpy as np


class BatchNorm1d:
    """
    Create your own mytorch.nn.BatchNorm1d!
    Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d class.
    Hint: Read all the expressions given in the writeup and be CAREFUL to re-check your code.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d forward.
        Note: The eval parameter indicate whether it's training phase or the inference phase of the problem.
        Check the values you need to recompute when eval = False.
        """
        self.Z = Z

        Z_transpose=np.transpose(Z)
        self.N = self.Z.shape[0]  # TODO: Calculate batch size
        Ones_N = np.ones((self.N, 1))
        self.M = np.sum(Z, axis=0, keepdims=True) / self.N  # TODO: Calculate mini-batch mean


        self.V = np.sum((Z - self.M) ** 2, axis=0, keepdims=True)/self.N # TODO: Calculate mini-batch variance

        if eval == False:
            # training mode
            self.NZ = (Z-Ones_N@self.M)/(Ones_N@np.sqrt(self.V+self.eps))  # TODO: Calculate the normalized input Ẑ
            self.BZ = (Ones_N@self.BW)*self.NZ +(Ones_N@self.Bb)  # TODO: Calculate the scaled and shifted for the normalized input Ẑ

            self.running_M = None  # TODO: Calculate running mean
            self.running_V = None  # TODO: Calculate running variance
        else:
            # inference mode
            self.NZ = (Z - Ones_N@self.running_M)/(Ones_N@np.sqrt(self.running_V + self.eps))   # TODO: Calculate the normalized input Ẑ using the running average for mean and variance
            self.BZ = (Ones_N@self.BW)*self.NZ+(Ones_N @ self.Bb)  # TODO: Calculate the scaled and shifted for the normalized input Ẑ

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d backward.
        """
        self.dLdBZ = dLdBZ
        self.dLdBb =np.sum(self.dLdBZ,axis=0,keepdims=True)  # TODO: Sum over the batch dimension.
        self.dLdBW =np.sum(self.dLdBZ*self.NZ,axis=0,keepdims=True)   # TODO: Scale gradient of loss wrt BatchNorm transformation by normalized input NZ.

        dLdNZ = self.dLdBZ*self.BW  # TODO: Scale gradient of loss wrt BatchNorm transformation output by gamma (scaling parameter).

        dLdV = -0.5*np.sum(dLdNZ*(self.Z-self.M)*((self.V+self.eps)**(-1.5)),axis=0,keepdims=True)  # TODO: Compute gradient of loss backprop through variance calculation.
        dNZdM = -((self.V+self.eps)**(-0.5))-0.5*(self.Z-self.M)*((self.V+self.eps)**(-1.5))*(-2*(np.sum(self.Z-self.M,axis=0,keepdims=True))/self.N)  # TODO: Compute derivative of normalized input with respect to mean.
        dLdM = np.sum(dLdNZ*dNZdM,axis=0,keepdims=True)  # TODO: Compute gradient of loss with respect to mean.

        dLdZ = dLdNZ*(((self.V+self.eps)**(-0.5)))+dLdV*(2*(self.Z-self.M)/self.N)+dLdM/self.N # TODO: Compute gradient of loss with respect to the input.
        return dLdZ  # TODO - What should be the return value?
