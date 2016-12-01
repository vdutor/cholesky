import tensorflow as tf
import numpy as np
import time
from numpy.random import randn
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline

# Autor: Dougal J. Sutherland
def cholesky_unblocked(X, size = None):
    if size is None:
        # need to have an explicit shape for X
        shape = X.get_shape()
        shape.assert_has_rank(2)
        m = shape[0].value
        assert m is not None
        assert shape[1].value == m
    else:
        m = size

    A = tf.matrix_band_part(X, -1, 0)
    cond = lambda j, A: j < m
    def body(j, A):
        s = A[j, :j]
        s = A[j, :j]
        s2 = tf.matmul(tf.expand_dims(s, 0), tf.expand_dims(s, 1))[0, 0]
        Ajj = tf.sqrt(A[j, j] - s2)

        col = A[j+1:m, j]
        col -= tf.matmul(A[j+1:m, :j], tf.expand_dims(A[j, :j], 1))[:, 0]
        col /= Ajj

        A = tf.concat(0, [
            A[:j],
            tf.concat(1, [
                A[j:, :j],
                tf.expand_dims(tf.concat(0, [tf.expand_dims(Ajj, 0), col]), 1),
                A[j:, j+1:]
            ])])
        A.set_shape([m, m])
        return j + 1, A
    return tf.while_loop( cond, body, (tf.constant(0, tf.int32), A))[1]


def add_top_padding(X, n, m):
    """
        Adds n rows of padding to the top of X.
        m equals the number of columns of X
    """
    padding = tf.zeros((n, m), dtype=X.dtype)
    return tf.concat(0, [padding, X])



# Author: Vincent Dutordoir
def cholesky_blocked(A, nb):
    """
        nb - block size, A.shape[0] % nb == 0
    """
    shape = A.get_shape()
    m = shape[0].value
    shape = tf.shape(A, out_type="int32")
    n = shape[0]
    # nb = 1 # tf.Variable(1, dtype="int32")
    j = tf.Variable(0, dtype="int32")

    # make A a lower triangular matrix
    A = tf.matrix_band_part(A, -1, 0)

    def body(A, j):
        A11 = tf.slice(A, [j, j], [nb, nb])
        A21 = tf.slice(A, [j + nb, j], [n - j - nb, nb])
        A22 = tf.slice(A, [j + nb, j + nb], [n - j - nb, n - j - nb])

        A11 = cholesky_unblocked(A11, nb)
        A21 = tf.matmul(A21, tf.matrix_inverse(tf.transpose(A11)))
        A22 = A22 - tf.matmul(A21, tf.transpose(A21))

        B_left = tf.concat(0, [A11, A21])
        B_right = add_top_padding(A22, nb, n - j - nb)
        B = tf.concat(1, [B_left, B_right])

        A_left = tf.slice(A, [0, 0], [n, j])
        A_right = add_top_padding(B, j, n - j)
        A = tf.concat(1, [A_left, A_right])

        # TODO remove this line
        A.set_shape([m, m])

        return A, j + nb

    def condition(A, j):
        return j < n

    return tf.while_loop(condition, body, [A, j])[0]


#
# main
#

def benchmark_blocked(block_sizes, matrix_size, plot):
    times = []
    BNs = []
    for BN in block_sizes:
        print "\nmatrix size: ", BN
        A = np.cov(randn(matrix_size, 3*matrix_size))
        A = tf.Variable(A, dtype="float64")

        A_chol_blocked = cholesky_blocked(A, BN)

        with tf.Session() as s:
            tf.initialize_all_variables().run()

            # tf version
            t0 = time.time()
            e = s.run(A_chol_blocked)
            t1 = time.time()
            print "time: ", t1 - t0
            times += [t1 - t0]

        BNs += [BN]

    if plot:
        fig = plt.figure(0)
        fig.canvas.set_window_title('Block size')
        plt.plot(BNs, times, '--o', color='r') # , label='tf')
        plt.xlabel('BN')
        plt.ylabel('time (s)')
        plt.show()


def benchmark_model(matrix_orders, plot, trace):
    t_blocked = []
    t_tf = []
    t_unblocked = []
    Ns = []
    for N in matrix_orders:
        print "\nmatrix size: ", N
        A = np.cov(randn(N, 3*N))
        A = tf.Variable(A, dtype="float64")

        A_chol = tf.cholesky(A)
        A_chol_unblocked = cholesky_unblocked(A)
        A_chol_blocked = cholesky_blocked(A, 100)
        # error_blocked = tf.reduce_sum(tf.pow(A_chol_blocked - A_chol, 2))
        # error_unblocked = tf.reduce_sum(tf.pow(A_chol_unblocked - A_chol, 2))

        with tf.Session() as s:
            tf.initialize_all_variables().run()

            # tf version
            run_metadata = tf.RunMetadata()
            t0 = time.time()
            e = s.run(A_chol)
                      # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                      # run_metadata=run_metadata)
            t1 = time.time()
            # print "time blocked version: {} -- error: {}".format(t1 - t0, e)
            print "time tf: ", t1 - t0
            t_tf += [t1 - t0]
            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('tf_cholesky.json', 'w') as f:
                # f.write(ctf)


            # blocked version
            run_metadata = tf.RunMetadata()
            t0 = time.time()
            e = s.run(A_chol_blocked)
                      # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                      # run_metadata=run_metadata)
            t1 = time.time()
            # print "time blocked version: {} -- error: {}".format(t1 - t0, e)
            print "time blocked: ", t1 - t0
            t_blocked += [t1 - t0]
            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('blocked_cholesky.json', 'w') as f:
                # f.write(ctf)

            # unblocked version
            # run_metadata = tf.RunMetadata()
            # t0 = time.time()
            # e = s.run(A_chol_unblocked)
                      # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                      # run_metadata=run_metadata)
            # t1 = time.time()
            # print "time unblocked version: {} -- error: {}".format(t1 - t0, e)
            # print "time unblocked: ", t1 - t0
            # t_unblocked += [t1 - t0]
            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('unblocked_cholesky.json', 'w') as f:
                # f.write(ctf)

        # Ns  += [N]

        # fig = plt.figure(0)
        # fig.canvas.set_window_title('GPU')
        # plt.plot(Ns, t_tf, color='g', label='tf')
        # plt.plot(Ns, t_blocked, color='r', label='blocked')
        # plt.plot(Ns, t_unblocked, color='b', label='unblocked')
        # # plt.title('Easy as 1, 2, 3')
        # plt.legend(loc='best')
        # plt.xlabel('N')
        # plt.ylabel('time (s)')
        # plt.show()

# ###########
# MAIN
# ###########

benchmark_blocked([1, 50, 100, 200, 400, 500], 2000, True)
