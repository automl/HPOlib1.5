import numpy as np


def get_rng(rng=None, self_rng=None):
    """ helper function to obtain RandomState.
    returns RandomState created from rng
    if rng is None returns self_rng if RandomState
    if self_rng is None initializes RandomState at random

    :param rng: int or RandomState
    :param self_rng: RandomState
    :return: RandomState
    """

    if rng is not None:
        return create_rng(rng)
    elif rng is None and self_rng is not None:
        return create_rng(self_rng)
    else:
        return np.random.RandomState()


def create_rng(rng):
    """ helper to create rng from RandomState or int
    :param rng: int or RandomState
    :return: RandomState
    """
    if rng is None:
        return np.random.RandomState()
    elif type(rng) == np.random.RandomState:
        return rng
    elif int(rng) == rng:
        return np.random.RandomState(rng)
    else:
        raise ValueError("%s is neither a number nor a RandomState. "
                         "Initializing RandomState failed")