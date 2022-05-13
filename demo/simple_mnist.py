import numpy
import matplotlib.pyplot

with open("../data/mnist/train.csv", 'r') as f:
    data_list = f.readlines()

all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28)) # 784文字を28×28の配列へ変換
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.savefig('number.png')
