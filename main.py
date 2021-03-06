import tensorflow as tf
import numpy as np
import time
from numpy.random import randn
from tensorflow.python.client import timeline
from cholesky import cholesky_blocked, cholesky_unblocked

import platform
if platform.node() == 'sumo-radar':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

#
# Benchmark functions
#

def positive_definite_tensor(N):
    """
        generates a positive difinite tensor of order N

    """
    A = np.cov(randn(N, 3*N))
    # print "\noriginal matrix (A):"
    # print A
    A = tf.Variable(A, dtype="float64")
    return A


def test_gradient():
    A = positive_definite_tensor(4)
    # A = tf.Variable(np.eye(5))
    tf_chol = tf.cholesky(A)
    grad_tf = tf.gradients(tf_chol, [A])[0]
    chol = cholesky_blocked(A, block_size=2)
    grad_blocked = tf.gradients(chol, [A])[0]
    # grad_tf = tf.gradients((A), [A])[0]

    with tf.Session() as s:
        tf.global_variables_initializer().run()
        grad_tf = s.run([A,tf_chol, grad_tf])
        tf.global_variables_initializer().run()
        grad_blocked = s.run([A,chol, grad_blocked])
        print "Tensorflow"
        print "Chol\n", grad_tf[1]
        print "Grads\n", grad_tf[2]
        print "Own implementation"
        print "Chol\n", grad_blocked[1]
        print "Grads\n", grad_blocked[2]


def test(N):
    """
        runs cholesky blocked and tf.cholesky on the same matrix
        of order N and prints the result

    """
    A = positive_definite_tensor(N)

    A_chol_blocked = cholesky_blocked(A, block_size=2)
    LLT = tf.matmul(A_chol_blocked, tf.transpose(A_chol_blocked))
    error = A - LLT
    A_chol = tf.cholesky(A)

    with tf.Session() as s:
        tf.global_variables_initializer().run()

        A_chol_blocked, LLT, error = s.run([A_chol_blocked, LLT, error])
        A_chol = s.run(A_chol)

        print "\nblocked decomposition (L): \n", A_chol_blocked
        print "\ntf decomposition (A): \n", A_chol
        print "\nLLT (L computed with blocked algorithma:)\n", LLT
        print "\nerror:\n", error


def benchmark_blocked(block_sizes, matrix_order, plot):
    """
        test the influence of block size on the execution time of cholesky_blocked

    """
    times = []
    for BN in block_sizes:
        print "\nmatrix size: ", BN
        A = positive_definite_tensor(matrix_order)

        A_chol_blocked = cholesky_blocked(A, BN)

        with tf.Session() as s:
            tf.initialize_all_variables().run()

            t0 = time.time()
            result = s.run(A_chol_blocked)
            t1 = time.time()
            print "time: ", t1 - t0
            times += [t1 - t0]

    if plot:
        plt.plot(block_sizes, times, '--o', color='r')
        plt.xlabel('BN')
        plt.ylabel('time (s)')
        plt.show()


def benchmark_implementations(matrix_orders, plot, trace):
    """
        Benchmarks tf.cholesk, cholesky_blocked and cholesky_unblocked on matrices
        with order 'matrix_orders'
        Note: when trace is True, plotting is not possible and visa versa

    """
    implementations = {'tf.cholesky': {'tensor_fn': tf.cholesky, 'durations': [], 'color': 'b'},
                       'cholesky_blocked': {'tensor_fn': cholesky_blocked , 'durations': [], 'color': 'r'}}
                       # 'unblocked': {'tensor_fn': cholesky_unblocked, 'durations': [], 'color': 'b'}}

    for N in matrix_orders:
        print "\nmatrix size: ", N
        A = positive_definite_tensor(N)

        with tf.Session() as s:
            for impl, data in implementations.iteritems():
                print impl
                tensor = tf.gradients(data['tensor_fn'](A), [A])
                tf.global_variables_initializer().run()

                if trace:
                    run_metadata = tf.RunMetadata()
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    s.run(tensor, options=options, run_metadata=run_metadata)

                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open(impl + '_cholesky.json', 'w') as f:
                        f.write(ctf)
                else:
                    t0 = time.time()
                    result = s.run(tensor)
                    t1 = time.time()
                    data['durations'].append(t1 - t0)
                    print data['durations'][-1]



    if not trace and plot:
        for impl, data in implementations.iteritems():
            plt.semilogy(matrix_orders, data['durations'], '--o', color=data['color'], label=impl)

        plt.grid(True)
        # plt.legend(loc='best')
        plt.xlabel('N')
        plt.ylabel('time (s)')
        plt.show()
        # plt.savefig('updated_cholesky.png')

#
# MAIN
#

benchmark_implementations([50,100], True, False)
