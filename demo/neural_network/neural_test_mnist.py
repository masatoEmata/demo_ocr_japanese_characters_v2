import pickle
import numpy
from neural_common import parse_label_digits, scaled_input

def test(n):
    train_path = "../../data/mnist/test.csv"
    score_card = []
    with open(train_path, 'r') as f:
        train_data_list = f.readlines()

    for record in train_data_list:
        true_label, digits = parse_label_digits(record)
        inputs = scaled_input(digits)
        final_outputs = n.query(inputs)
        prd_label = numpy.argmax(final_outputs)
        print(f'true label: {true_label}')
        print(f'predicted label: {prd_label}')
        if prd_label == true_label:
            score_card.append(1)
        else:
            score_card.append(0)
        # print(f'outputs: {final_outputs}')
    print('sum: ', sum(score_card))
    print('len: ', len(score_card))
    print('sum(score_card)/len(score_card): ', sum(score_card)/len(score_card))

    score_card_array = numpy.asarray(score_card)
    print('score_card_array.sum(): ', score_card_array.sum())
    print('score_card_array.size: ', score_card_array.size)
    print('score_card_array.sum()/score_card_array.size: ', score_card_array.sum()/score_card_array.size)
    # print(f'score: {score}')
    return n


if __name__ == '__main__':
    from neural_network import NeuralNetwork
    from neural_train_mnist import train

    output_nodes = 10
    input_nodes = 784
    hidden_nodes = 200
    learning_rate = 0.1
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    epoch = 200
    n = train(n, 1)

    test(n)
