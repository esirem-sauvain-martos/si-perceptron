import csv
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_count, epoch_count, learning_rate, delta):
        self._input_count = input_count
        self._epoch_count = epoch_count
        self._learning_rate = learning_rate
        self._delta = delta
        self._weigths = [0 for _ in range(input_count)]
        self._biais = 1
        self._biais_weigth = 0
        self._error_values = []

    def predict(self, input_A, input_B):
        computed_res = input_A * self._weigths[0] + input_B * self._weigths[1] + self._biais * self._biais_weigth
        return 1 if computed_res > 0 else 0

    def train(self, inputs, expected_outputs):
        # for _ in range(self._epoch_count):
        keep_training = True
        last_error_rate_average = 1000

        while keep_training:
            epoch_error_value = 0
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i][0], inputs[i][1])

                for j in range(len(self._weigths)):
                    new_weight = self._weigths[j] + self._learning_rate * (expected_outputs[i] - prediction) * inputs[i][j]
                    self._weigths[j] = new_weight

                self._biais_weigth = self._biais_weigth + self._learning_rate * (expected_outputs[i] - prediction) * 1

                epoch_error_value += 0.5 * ((prediction - expected_outputs[i]) ** 2)
            self._error_values.append(epoch_error_value)

            new_error_rate_average = sum(self._error_values) / len(self._error_values)
            if abs(new_error_rate_average - last_error_rate_average) < self._delta:
                keep_training = False
            last_error_rate_average = new_error_rate_average

    def show_error_graph(self):
        plt.plot(self._error_values)
        plt.show()

    def save_weight(self):
        filename = 'perceptron.csv'
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header_line = ['w1', 'w2', 'w biais']
            csv_writer.writerow(header_line)
            csv_writer.writerow(self._weigths + [self._biais_weigth])

    def load_weight(self):
        filename = 'perceptron.csv'
        with open(filename, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for line in csv_reader:
                weights = line
            for i in range(len(weights) - 1):
                self._weigths[i] = weights[i]
            self._biais_weigth = weights[-1]

        print(self)

    def __str__(self):
        return "Perceptron [weights = {}, biais weight = {}]".format(self._weigths, self._biais_weigth)

