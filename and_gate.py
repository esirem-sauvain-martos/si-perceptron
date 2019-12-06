from perceptron import Perceptron

def main():
    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    outputs = [
        0,
        0,
        0,
        1
    ]

    perceptron = Perceptron(2, 7, 0.01, 0.039)
    perceptron.train(inputs, outputs)

    for possible_input in inputs:
        print(perceptron.predict(possible_input[0], possible_input[1]))

    perceptron.show_error_graph()
    
    # perceptron.save_weight()
    # perceptron.load_weight()



if __name__ == "__main__":
    main()
