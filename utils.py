import numpy as np
import tensorflow as tf
import keras
from numba import jit

def rmsre(y_true, y_pred, eps=1e-8):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square((y_true - y_pred) / (y_true + eps)), axis=-1))

def LB_stencil():

    ###########################################################
    # D2Q9 stencil
    Q = 9
    c = np.zeros((Q, 2), dtype=np.int32)
    w = np.zeros(Q)    

    cs2     = 1./3.
    qorder  = 2

    c[0, 0] =  0;  c[0, 1] =  0; w[0] = 4./9.
    c[1, 0] =  1;  c[1, 1] =  0; w[1] = 1./9.
    c[2, 0] =  0;  c[2, 1] =  1; w[2] = 1./9.
    c[3, 0] = -1;  c[3, 1] =  0; w[3] = 1./9.
    c[4, 0] =  0;  c[4, 1] = -1; w[4] = 1./9.
    c[5, 0] =  1;  c[5, 1] =  1; w[5] = 1./36.
    c[6, 0] = -1;  c[6, 1] =  1; w[6] = 1./36.
    c[7, 0] = -1;  c[7, 1] = -1; w[7] = 1./36.
    c[8, 0] =  1;  c[8, 1] = -1; w[8] = 1./36.

    ###########################################################
    # Function for the calculation of the equilibrium
    @jit
    def compute_feq(feq, rho, ux, uy, c, w):

        uu = (ux**2 + uy**2)*(1./cs2)

        for ip in range(Q):

            cu = (c[ip, 0]*ux[:,:]  + c[ip, 1]*uy[:,:] )*(1./cs2)

            feq[:, :, ip] = w[ip]*rho*(1.0 + cu + 0.5*(cu*cu - uu) )

        return feq

    ###########################################################

    return c, w, cs2, compute_feq


def LBrot90(f, k=1):
    return tf.concat([f[:, 0, None], tf.roll(f[:, 1:5], k, axis=-1), tf.roll(f[:, 5:], k, axis=-1)], axis=-1)


def LBmirror(f):
    return tf.concat(
        [
            f[:, 0, None],
            f[:, 1, None],
            f[:, 4, None],
            f[:, 3, None],
            f[:, 2, None],
            f[:, 8, None],
            f[:, 7, None],
            f[:, 6, None],
            f[:, 5, None],
        ],
        axis=-1,
    )


@keras.saving.register_keras_serializable(package="lbm")
class D4Symmetry(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):

        y = [
            x,
            LBrot90(x, k=1),
            LBrot90(x, k=2),
            LBrot90(x, k=3),
            LBmirror(x),
            LBmirror(LBrot90(x, k=1)),
            LBmirror(LBrot90(x, k=2)),
            LBmirror(LBrot90(x, k=3)),
        ]

        return y


@keras.saving.register_keras_serializable(package="lbm")
class D4AntiSymmetry(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):

        y = [
            x[0],
            LBrot90(x[1], k=-1),
            LBrot90(x[2], k=-2),
            LBrot90(x[3], k=-3),
            LBmirror(x[4]),
            LBrot90(LBmirror(x[5]), k=-1),
            LBrot90(LBmirror(x[6]), k=-2),
            LBrot90(LBmirror(x[7]), k=-3),
        ]

        return y


@keras.saving.register_keras_serializable(package="lbm")
class AlgReconstruction(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, fpre, fpred):

        df = fpred - fpre

        df2 = -(df[:, 0] + 2 * df[:, 3] + df[:, 4] + 2 * df[:, 6] + 2 * df[:, 7])
        df5 = 0.5 * (df[:, 0] + 3 * df[:, 3] + 2 * df[:, 4] + 2 * df[:, 6] + 4 * df[:, 7] - df[:, 1])
        df8 = -0.5 * (df[:, 0] + df[:, 1] + df[:, 3] + 2 * df[:, 4] + 2 * df[:, 7])

        df = tf.concat(
            [
                df[:, 0, None],
                df[:, 1, None],
                df2[:, None],
                df[:, 3, None],
                df[:, 4, None],
                df5[:, None],
                df[:, 6, None],
                df[:, 7, None],
                df8[:, None],
            ],
            axis=-1,
        )

        res = fpre + df

        return res
