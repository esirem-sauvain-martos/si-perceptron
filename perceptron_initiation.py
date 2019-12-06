import numpy as np
import matplotlib.pyplot as plt


def T(x):
    return 0 if x <= 0 else 1


def main():
    possible_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    possible_outputs = [
        0,
        0,
        0,
        1
    ]

    results = np.zeros((10, 10))

    for w1 in range(-5, 5):
        for w2 in range(-5, 5):
            w1_w2_res = 0
            for k in range(4):
                res = T(w1 * possible_inputs[k]
                        [0] + w2 * possible_inputs[k][1])
                error = 0.5 * (res - possible_outputs[k]) ** 2
                w1_w2_res += error
            results[w1 + 5][w2 + 5] = w1_w2_res

    print(results)
    plt.imshow(results, cmap='winter')
    plt.show()


if __name__ == "__main__":
    main()
