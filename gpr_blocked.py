import time
import numpy as np
import GPflow
import tensorflow as tf
from cholesky import cholesky_blocked2

class GPR_blocked(GPflow.gpr.GPR):
    def __init__(self, X, Y, kern):
        GPflow.gpr.GPR.__init__(self, X, Y, kern)
        self.num_samples = X.shape[0] # cholesky_blocked needs to know the matrix size

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + GPflow.tf_wraps.eye(tf.shape(self.X)[0]) * self.likelihood.variance
        L = cholesky_blocked2(K, matrix_order=self.num_samples) # <-- New cholesky
        m = self.mean_function(self.X)

        return GPflow.densities.multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + GPflow.tf_wraps.eye(tf.shape(self.X)[0]) * self.likelihood.variance
        L = cholesky_blocked2(K, matrix_order=self.num_samples) # <-- New cholesky
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

def data():
    dataset = load_boston()

    # shuffle dataset
    np.random.seed(0)
    p = np.random.permutation(len(dataset.data))
    data = dataset.data[p]
    target = dataset.target[p]

    X_train = data[:10]
    Y_train = target.reshape((-1,1))[:10]

    return X_train, Y_train

# # # # #
# Main
# # # # #
N = 1000
maxiter = 1
X_train = np.linspace(-10,10,N).reshape(-1,1)
Y_train = np.sinc(X_train)
optimizer = tf.train.GradientDescentOptimizer(0.1)

# gpr model
kernel = GPflow.kernels.RBF(X_train.shape[1], ARD=True)
gpr_model = GPflow.gpr.GPR(X_train, Y_train, kern=kernel)

start_time = time.time()
gpr_model.optimize(optimizer, maxiter=maxiter)
time_train = time.time() - start_time
print "time for gpr model ", time_train

# gpr blocked model
kernel_blocked = GPflow.kernels.RBF(X_train.shape[1], ARD=True)
gpr_blocked_model = GPR_blocked(X_train, Y_train, kern=kernel_blocked)

start_time = time.time()
gpr_blocked_model.optimize(optimizer, maxiter=maxiter)
time_train_blocked = time.time() - start_time
print "time for gpr blocked  model ", time_train_blocked
