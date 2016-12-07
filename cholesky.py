import tensorflow as tf

# Author: Dougal J. Sutherland
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
        m equals the number of columns of X.
    """
    padding = tf.zeros((n, m), dtype=X.dtype)
    return tf.concat(0, [padding, X])


def cholesky_blocked(A, matrix_order = None, block_size = 200):
    """
        parameters:
            A  -- 2D tensor, must have a shape
            nb -- block size
    """
    nb = block_size
    if matrix_order is not None:
        n = matrix_order
    else:
        n = A.get_shape()[0].value

    if n is None:
        print "Error. Size of matrix A must be known"

    nb2 = n % nb

    # make A a lower triangular matrix
    A = tf.matrix_band_part(A, -1, 0)

    def body(A, j):
        def f1(A):
            B11 = tf.slice(A, [j, j], [nb, nb])
            B21 = tf.slice(A, [j + nb, j], [n - j - nb, nb])
            B22 = tf.slice(A, [j + nb, j + nb], [n - j - nb, n - j - nb])

            B11 = cholesky_unblocked(B11, nb)
            B21 = tf.transpose(tf.matrix_triangular_solve(B11, tf.transpose(B21), lower=True))
            B22 = B22 - tf.matmul(B21, tf.transpose(B21))

            B_left = tf.concat(0, [B11, B21])
            B_right = add_top_padding(B22, nb, n - j - nb)
            B = tf.concat(1, [B_left, B_right])

            A_left = tf.slice(A, [0, 0], [n, j])
            A_right = add_top_padding(B, j, n - j)
            A = tf.concat(1, [A_left, A_right])
            return A

        def f2(A):
            B11 = tf.slice(A, [j, j], [nb2, nb2])
            B =  cholesky_unblocked(B11, nb2)

            A_left = tf.slice(A, [0, 0], [n, j])
            A_right = add_top_padding(B, j, n - j)
            A = tf.concat(1, [A_left, A_right])
            return A

        A = tf.cond(j + nb > n, lambda: f2(A), lambda: f1(A))

        # TODO remove this line
        A.set_shape([n, n])

        return A, j + nb

    def condition(A, j):
        return j < n

    return tf.while_loop(condition, body, [A, 0])[0]


# TODO: doesn't work yet
def gradient_cholesky_blocked(A):
    chol = cholesky_blocked(A, block_size=2)
    return tf.gradients(chol, [A])[0]


