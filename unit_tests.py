import tensorflow as tf
import numpy as np
from cholesky import cholesky_blocked
import unittest
from numpy.random import randn

def positive_definite_tensor(N):
    """
        generates a positive difinite tensor of order N

    """
    A = np.cov(randn(N, 3*N))
    A = tf.Variable(A, dtype="float64")
    return A

class TestCholeskyMethods(unittest.TestCase):
    def setUp(self):
        self.A = positive_definite_tensor(100)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def test_chol_1(self):
        """
            Compares cholesky implementation with the tf.cholesky.
            In this test block_size divides matrix size, i.e. matrix_size % block_size == 0
        """
        A_chol_blocked = cholesky_blocked(self.A, block_size=20)
        A_chol_tf = tf.cholesky(self.A)
        error = self.session.run(tf.reduce_sum(tf.square(A_chol_blocked - A_chol_tf)))
        self.assertTrue(error < 1.E-5)

    def test_chol_2(self):
        """
            Compares cholesky implementation with the tf.cholesky.
            In this test block_size does not divides matrix size, i.e. matrix_size % block_size != 0
        """
        A_chol_blocked = cholesky_blocked(self.A, block_size=19)
        A_chol_tf = tf.cholesky(self.A)
        error = self.session.run(tf.reduce_sum(tf.square(A_chol_blocked - A_chol_tf)))
        self.assertTrue(error < 1.E-5)

    def test_chol_grad(self):
        """
            Test gradient calculations
        """
        grad_tf = tf.gradients(tf.cholesky(self.A), [self.A])[0]
        grad_blocked = tf.gradients(cholesky_blocked(self.A, block_size=20), [self.A])[0]
        error = self.session.run(tf.reduce_sum(tf.square(grad_blocked - grad_blocked)))
        self.assertTrue(error < 1.E-5)


if __name__ == '__main__':
    unittest.main()
