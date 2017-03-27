import numpy as np
import matplotlib.pyplot as plt
from NNDeepLearning import NeuralNetwork

def genData(N):
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D))
    y = np.zeros(N*K, dtype='uint8')
    for j in xrange(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      y[ix] = j
    T = np.zeros((y.shape[0],y.max()+1))
    for i in xrange(len(y)-1):
        T[i, y[i]] = 1    
    return X,T,y
    

nTrain = 1000
trainX,trainT,trainY = genData(nTrain)    
plt.scatter(trainX[:, 0], trainX[:, 1], c=trainY, s=40, cmap=plt.cm.Spectral)


nTest=100
testX,testT,testY = genData(nTest)   



NN = NeuralNetwork(trainX,trainT,[5,5],batchSize = 100)
NN.learningRate = 2e-3
epochs = 5000
for i in range(1,epochs):
    NN.train()
    if i % 400 == 0:
        Prediction = NN.forwardProp(trainX) 
        TestPrediction = NN.forwardProp(testX)  
        print 'Training Cost: {0:.3f} Training classification rate {1:.1f}%'.format(NN.cost[-1],100*NN.classificationRate(trainT,Prediction))
        print 'Test Cost: {0:.3f} Test classification rate {1:.1f}%\n'.format((testT * np.log(TestPrediction)).sum() / len(testY),100*NN.classificationRate(testT,TestPrediction))  
        
plt.figure()
plt.plot(NN.cost)

N=300
X,T,Y = genData(N)   
TestPrediction = NN.forwardProp(X)  
classificationRate = NN.classificationRate(T,TestPrediction)
