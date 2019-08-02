import mnist_loader
import network

training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
#print("training data")
#print(type(training_data))
#print(len(training_data))
#print(type(training_data[0]))
#print(training_data[0][0].shape)
#print(training_data[0][1].shape)

net = network.Network([784,30,10])
net.SGD(training_data,30,10,0.01,test_data=test_data)