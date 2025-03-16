import numpy as np


def test():
    matA = np.zeros((4, 4))
    matB = np.zeros((4, 4))
    for i in range(16):
        matA[i // 4, i % 4] = i + 1
        matB[i // 4, i % 4] = i + 1

    print(matA @ matB)


if __name__ == "__main__":
    test()
