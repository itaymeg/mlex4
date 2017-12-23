import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC, SVC
import sklearn.preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import lib
import operator


# data
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = np.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = np.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos)*2-1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_unscaled = data[60000+test_idx, :].astype(float)
test_labels = (labels[60000+test_idx] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


train_data = lib.normalize(train_data)
validation_data = lib.normalize(validation_data)
test_data = lib.normalize(test_data)

#

def sgd(samples, labels, c, lr, T):
    w = np.zeros(samples.shape[1]).astype(np.double)
    for i in range(T):
        idx = np.random.randint(0, samples.shape[0])
        current_lr = lr/(i+1)
        if labels[idx] * w.dot(samples[idx]) < 1: # yi*w*xi
            w = (1 - current_lr)*w + current_lr*c*labels[idx]*samples[idx]
        else:
            w = (1- current_lr) * w
    return w

def calc_accuracy(w, data, labels):
    correct = 0
    for i, sample in enumerate(data):
        prediction = 0
        if w.dot(sample) >= 0:
            prediction = 1
        else:
            prediction = -1
        if prediction == labels[i]:
            correct += 1 
    return float(correct) / float(data.shape[0])

#a
    
T = 1000
c = 1
corrects = {}
for _ in range(10):
    for lr in [10**power for power in range(-5, 5, 1)]:
            #train
        w = sgd(train_data, train_labels, c, lr, T)
            #validate
        correct = calc_accuracy(w, validation_data, validation_labels)
        if lr not in corrects:
            corrects[lr] = 0
        corrects[lr] += correct
for lr in corrects:
    corrects[lr] /= 10 # average

sorted_correct = sorted(corrects.items(), key=operator.itemgetter(1), reverse=True)
best_lr = sorted_correct[0][0]
sorted_by_lr = sorted(corrects.items(), key=operator.itemgetter(0))
xsys = zip(*sorted_by_lr)
xs, ys = xsys[0], xsys[1]
plt.semilogx(xs, ys, marker=(4,0))
plt.xlabel('n0')
plt.ylabel('Average Accuracy')
plt.title('Accuracy as a function of n0')
plt.savefig('1_a.jpg')
plt.show()
    

#b
    
T = 1000
lr = best_lr
corrects = {}
for _ in range(10):
    for c in [10**power for power in range(-5, 5, 1)]:
            #train
        w = sgd(train_data, train_labels, c, lr, T)
            #validate
        correct = calc_accuracy(w, validation_data, validation_labels)
        if c not in corrects:
            corrects[c] = 0
        corrects[c] += correct
for c in corrects:
    corrects[c] /= 10 # average

sorted_correct = sorted(corrects.items(), key=operator.itemgetter(1), reverse=True)
best_c = sorted_correct[0][0]
sorted_by_c = sorted(corrects.items(), key=operator.itemgetter(0))
xsys = zip(*sorted_by_c)
xs, ys = xsys[0], xsys[1]
plt.semilogx(xs, ys, marker=(4,0))
plt.xlabel('c')
plt.ylabel('Average Accuracy')
plt.title('Accuracy as a function of c')
plt.savefig('1_b.jpg')
plt.show()


#c

c = 1
lr = 1
T = 20000
w = sgd(train_data, train_labels, c, lr, T)
plt.imshow(w.reshape(28, 28))
plt.title('Image representing the W')
plt.savefig('1_c.jpg')
plt.show()

#d
    
c = 1
lr = 1
T = 20000
w = sgd(train_data, train_labels, c, lr, T)
acc = calc_accuracy(w, test_data, test_labels)
print('Accuracy on test Set : ', acc)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    