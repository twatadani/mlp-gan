import numpy as np


def create_random():
    return np.random.rand(1, 10)


for i in range(100):
    ar = create_random()
    print(ar)
