from perceptron import Perceptron

def main():
    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    outputs = [
        [0],
        [0],
        [0],
        [1]
    ]

    perceptron = Perceptron(2, 1, 0.01, 500)

    # perceptron.load_weight()
    perceptron.train(inputs, outputs)

    for possible_input in inputs:
        possible_input.append(1)
        print(perceptron.predict(possible_input, 0))

    perceptron.save_weight()


if __name__ == "__main__":
    main()
