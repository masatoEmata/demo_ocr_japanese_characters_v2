from sklearn import datasets
digits = datasets.load_digits()


# ■ データ概要確認
# print('== img data ==')
# print(digits.images[0])
# print(digits.images.shape)

# print('== n1 img data ==')
# print(digits.data[0])
# print(digits.data.shape)

# print('== label ==')
# print(digits.target)
# print(digits.target.shape)

# == img data ==
# [[ 0.  0.  5. 13.  9.  1.  0.  0.]
#  [ 0.  0. 13. 15. 10. 15.  5.  0.]
#  [ 0.  3. 15.  2.  0. 11.  8.  0.]
#  [ 0.  4. 12.  0.  0.  8.  8.  0.]
#  [ 0.  5.  8.  0.  0.  9.  8.  0.]
#  [ 0.  4. 11.  0.  1. 12.  7.  0.]
#  [ 0.  2. 14.  5. 10. 12.  0.  0.]
#  [ 0.  0.  6. 13. 10.  0.  0.  0.]]
# (1797, 8, 8)
# == n1 img data ==
# [ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
#  15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#   0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#   0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
# (1797, 64)
# == label ==
# [0 1 2 ... 8 9 8]
# (1797,)



# ■ 画像＆ラベルの表示
# imgs = digits.images
# lbls = digits.target
# for i in range(10):
#     plt.subplot(2, 5, i + 1 ) # 描画配置
#     plt.title(f'label: {lbls[i]}')
#     plt.imshow(imgs[i], cmap='Greys')
#     plt.axis('off')
# plt.show()




from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# データ準備
digits = datasets.load_digits()
x_train, x_test, t_train, t_test = train_test_split(digits.data, digits.target)

# モデリング
clf = svm.SVC()
clf.fit(x_train, t_train)
y_test = clf.predict(x_test)

# 評価
correct_rate = metrics.classification_report(t_test, y_test)
metrics_report = metrics.confusion_matrix(t_test, y_test)
print(correct_rate)
print(metrics_report)
## 目視
imgs = digits.images[:10]
lbls_prd = clf.predict(digits.data[:10])
for i in range(10):
    plt.subplot(2, 5, i + 1 ) # 描画配置
    plt.title(f'guess: {lbls_prd[i]}')
    plt.imshow(imgs[i], cmap='Greys')
    plt.axis('off')
plt.show()