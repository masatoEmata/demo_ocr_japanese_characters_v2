import glob
from numpy import record
import numpy
import imageio
import matplotlib.pyplot as plt
import pickle

with open('../data/train_model.pickle', mode='wb') as f:
    n = pickle.load(f)

my_own_dataset = []
own_img_path = '.../....?.png'
for img_file_name in glob.glob(own_img_path):
    label = int(img_file_name[-5:-4])
    print('loading...', img_file_name)
    img_array = imageio.imread(img_file_name, as_gray=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)
    my_own_dataset.append(record)

item = 0
plt.imshow(my_own_dataset[item][1:].reshape(28, 28), cmap='Greys', interpolation='None')

correct_label = my_own_dataset[item][0]
inputs = my_own_dataset[item][1:]


outputs = n.query(inputs)
prd_label = numpy.argmax(outputs)
print(f'network says: {prd_label}')
