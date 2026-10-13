"""Working single-tissue model plugin; copy and edit for another single-input ODE.

state is [N, STATE_SIZE], parameters [N, P], input_value [N].
rhs returns [N, STATE_SIZE]; observe returns [N]. Time is in minutes.
Zero initial states are used. Use NumPy operations that preserve the batch axis.
"""
import numpy as np

PARAMETER_NAMES = ['K1', 'k2', 'Vb']
STATE_SIZE = 1


def rhs(t, state, parameters, input_value):
    K1, k2, _ = parameters.T
    return (K1*input_value-k2*state[:, 0])[:, None]


def observe(state, parameters, input_value):
    vb = parameters[:, 2]
    return vb*input_value+(1-vb)*state[:, 0]
