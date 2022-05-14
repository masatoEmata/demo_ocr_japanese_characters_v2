from sklearn.model_selection import learning_curve
import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 入力層→隠れ層へのリンクの重み（w_ih）
        self.w_ih = numpy.random.normal(0.0, pow(self.inodes, -1/2), (self.hnodes, self.inodes))
        # 隠れ層→出力層へのリンクの重み（w_ho）
        self.w_ho = numpy.random.normal(0.0, pow(self.hnodes, -1/2), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # 第1の処理：与えられた訓練データの出力を得る処理
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 真値リストを行列に変換
        targets = numpy.array(targets_list, ndmin=2).T

        # 隠れ層に入ってくる信号の計算
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        # 隠れ層から出る信号に活性化関数を作用させる
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        # 出力層から出る最終的な出力に活性化関数を作用させる
        final_outputs = self.activation_function(final_inputs)

        # 第2の処理：訓練データの出力値と真の出力値（正解の値）を比較して、その差を用いてリンクの重みを修正していく処理
        # 誤差の比較
        output_errors = targets - final_outputs
        # 重みの更新
        # 隠れ層と出力層間のリンクの重みの更新
        self.w_ho += self.lr * numpy.dot( \
            (output_errors * final_outputs * (1.0 - final_outputs)), \
            numpy.transpose(hidden_outputs))
        # 入力層と隠れ層間のリンクの重みの更新
        # 誤差逆伝播
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)
        self.w_ih += self.lr * numpy.dot( \
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), \
            numpy.transpose(inputs))

    def query(self, inputs_list):
        # 入力リストを行列に変換(.Tを使って転置している)
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 入力信号が隠れ層に入ってくるまでの計算
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        # 隠れ層からの信号にシグモイド関数を作用させて出力層へ渡す
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        # 出力層からの信号にシグモイド関数を作用させて最終的な出力とする
        final_outputs = self.activation_function(final_inputs)

        # 関数の戻り値として最終的な出力結果を返す
        return final_outputs


if __name__ == '__main__':
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(n.query([1.0, 0.5, -0.5]))
# https://ruby-de-free.net/wp/hand-painted-numeric-judgment-ai-program/