import numpy as np
import matplotlib.pyplot as plt
from NNDeepLearning import NeuralNetwork
import signal
from forward2Hidden import getData
import pandas as pd
run = True

def signal_handler(signal, frame):
    global run
    print "exiting"
    run = False

signal.signal(signal.SIGINT, signal_handler)

#X,Y,Xtest,Ytest = getData(pd.read_json('train.json'))
Y = train_Y.astype('int32')
T = np.zeros((len(train_X), 3)).astype('int32')
for i in xrange(len(train_X)):
    T[i, Y[i]] = 1  


Xtrain = train_X
Ttrain = T
NN = NeuralNetwork(Xtrain,Ttrain,[10]*1,batchSize = 200,saveInterval = 100)
NN.learningRate = 0.02
i =0
while run:
    i +=1
    NN.train()
    if i % 100 == 0:
        Prediction = NN.forwardProp(Xtrain)    
        print "Cost is:", NN.cost[-1], "Classification Rate:", NN.classificationRate(T,Prediction), "Learning rate:",NN.learningRate
        print "test cr", np.mean(np.argmax(NN.forwardProp(test_X),axis=1) == test_Y.astype('int32'))
plt.plot(NN.cost)


#TestPrediction = NN.forwardProp(X[trainRatio:])  
#classificationRate = NN.classificationRate(T[trainRatio:],TestPrediction) 
Prediction = NN.forwardProp(Xtest)