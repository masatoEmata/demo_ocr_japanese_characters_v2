import numpy
from neural_network import NeuralNetwork
import dill as pickle
from neural_common import parse_label_digits, scaled_input


def train(n: NeuralNetwork, epoch):
    train_path = "../../data/mnist/train.csv"

    with open(train_path, 'r') as f:
        train_data_list = f.readlines()

    for e in range(epoch):
        print(f'Epoch {e+1}')
        for record in train_data_list:
            label, digits = parse_label_digits(record)
            inputs = scaled_input(digits)
            targets = numpy.zeros(n.onodes) + 0.01
            targets[int(label)] = 0.99
            n.train(inputs, targets)
    return n


if __name__ == '__main__':
    output_nodes = 10
    input_nodes = 784
    hidden_nodes = 200
    learning_rate = 0.1
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(f'w_ho: {n.w_ho}')
    epoch = 10
    n = train(n, 1)
    print(f'w_ho trained: {n.w_ho}')

    # 失敗。0バイトで書き込まれる
    # with open('../data/train_model.pickle', mode='wb') as f:
    #     pickle.dump(n, f)
