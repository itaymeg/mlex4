import data
import network

training_data, test_data = data.load(train_size=50000,test_size=10000)
net = network.Network([784, 40, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)