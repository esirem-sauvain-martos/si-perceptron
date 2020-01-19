import numpy as np
import csv
import math
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_inputs, nb_outputs, learning_rate, epochs, prediction_type = 0, biais = 1):
        self._inputs_count = nb_inputs + 1
        self._outputs_count = nb_outputs
        self._learning_rate = learning_rate
        self._prediction_type = prediction_type
        self._epochs = epochs
        self._weights = np.zeros((self._outputs_count, self._inputs_count))
        self._error_values = []
        self._biais = biais

    def predict(self, input_set, output_nb):
        res = np.dot(input_set, self._weights[output_nb])

        if self._prediction_type == 0:
            return 0 if res <= 0 else 1
        elif self._prediction_type == 1:
            return 0 if res <= 0 else res

    def train(self, all_inputs, expected_outputs):
        for epoch in range(self._epochs):
            for output_nb in range(self._outputs_count):
                for i, input_set in enumerate(all_inputs):
                    local_input_set = input_set[:]
                    local_input_set.append(self._biais)
                    self.update_weights(local_input_set, 
                        expected_outputs[i], output_nb)

    def update_weights(self, input_set, expected_output, output_nb):
        weights = self._weights[output_nb]
        prediction = self.predict(input_set, output_nb)

        for i, weight in enumerate(weights):
            new_weight = weights[i] + self._learning_rate * \
             (expected_output[output_nb] - prediction) * input_set[i]
            weights[i] = new_weight

        epoch_error_value = (prediction - expected_output[output_nb]) ** 2

        return epoch_error_value

    def show_error_graph(self):
        plt.plot(self._error_values)
        plt.show()

    def save_weight(self):
        filename = 'perceptron.csv'
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header_line = ['output', 'w1', 'w2', 'w biais']
            csv_writer.writerow(header_line)
            for k, output_weights in enumerate(self._weights):
                output_weights = np.append([k], output_weights)
                csv_writer.writerow(output_weights)

    def load_weight(self):
        filename = 'perceptron.csv'
        with open(filename, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            outputs_weights = []
            for l, line in enumerate(csv_reader):
                if l != 0:
                    outputs_weights.append(line)

            weights = []
            for output_weights in outputs_weights:
                weights.append(list(map(lambda x: float(x), output_weights[1:])))

            self._weights = weights.copy()
