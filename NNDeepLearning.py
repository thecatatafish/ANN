import numpy as np
'''
Neural network using sigmoid activation function and gradient decent
back probiation. Arbitrary number of hidden layers and hidden units implemented
'''
class NeuralNetwork:
    
    def __init__(self,X,Labels,nHiddenNodes,batchSize=100,saveInterval=100):
        self.X = X
        self.cost = []
        self.learningRate = 1e-5
        self.Labels = Labels
        self.nHiddenLayers = len(nHiddenNodes)
        self.nHiddenNodes = nHiddenNodes 
        self.nEpochs = 0
        self.saveInterval = saveInterval
        self.W = {} ; self.b = {} ; self.Z = {}
        self.nSamples,self.nFeatures = X.shape
        self.nClasses = Labels.shape[1]
        
        self.batchSize = int(batchSize)
        self.nBatches =   self.nSamples / self.batchSize
        self._reminder =  self.nSamples % self.batchSize
        
        self.W['W0'] = np.random.rand(self.nFeatures,self.nHiddenNodes[0]) 
        self.b['b0'] = np.zeros(self.nHiddenNodes[0])
        
        for layer in range(1,self.nHiddenLayers):
            self.W['W' + str(layer)] = np.random.rand(self.nHiddenNodes[layer-1],self.nHiddenNodes[layer]) 
            self.b['b' + str(layer)] = np.zeros(self.nHiddenNodes[layer])      

        self._lastHiddenLayer = 'Z'+str(self.nHiddenLayers)
        self._lastWeightMatrix = 'W'+ str(self.nHiddenLayers)
        self._lastBiasArray = 'b'+ str(self.nHiddenLayers)
        self.W[self._lastWeightMatrix] = np.random.rand(self.nHiddenNodes[-1],self.nClasses)
        self.b[self._lastBiasArray] = np.zeros(self.nClasses)
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    def softmax(self,Y):
        expA = np.exp(Y - np.max(Y))
        return expA/np.sum(expA,axis=1,keepdims = True)    
        
    def costFunction(self,T,Y):
        return (T * np.log(Y)).sum() / self.batchSize
    
    def classificationRate(self,T,Predictions):
        return np.mean( np.argmax(T,axis=1) == np.argmax(Predictions,axis=1) )
        
    def _forwardProp(self,X):
       self.Z["Z1"] = self.sigmoid(X.dot(self.W["W0"])+self.b["b0"])
       for layer in range(2,self.nHiddenLayers+1):
           prevLayer = str(layer-1)
           self.Z["Z" + str(layer)] = self.sigmoid(self.Z["Z" + prevLayer].dot(self.W["W" + prevLayer])+self.b["b" + prevLayer])
       Predictions = self.softmax(self.Z[self._lastHiddenLayer].dot(self.W[self._lastWeightMatrix]) + self.b[self._lastBiasArray])
       return Predictions 
       
    def forwardProp(self,X):
        Z={}
        Z["Z1"] = self.sigmoid(X.dot(self.W["W0"])+self.b["b0"])
        for layer in range(2,self.nHiddenLayers+1):
            prevLayer = str(layer-1)
            Z["Z" + str(layer)] = self.sigmoid(Z["Z" + prevLayer].dot(self.W["W" + prevLayer])+self.b["b" + prevLayer])
        Predictions = self.softmax(Z[self._lastHiddenLayer].dot(self.W[self._lastWeightMatrix]) + self.b[self._lastBiasArray])
        return Predictions 
       
    def backProp(self,X,T):
        Predictions = self._forwardProp(X)
        dJ_dY = T-Predictions
        dW = {}; db = {}; dZ = {}
        dW[self._lastWeightMatrix] = self.Z[self._lastHiddenLayer].T.dot(dJ_dY)
        db[self._lastBiasArray] = dJ_dY.sum(axis=0)
        
        dZ[self._lastHiddenLayer] = dJ_dY.dot(self.W[self._lastWeightMatrix].T)* self.Z[self._lastHiddenLayer]*(1-self.Z[self._lastHiddenLayer])
        
        for layer in reversed(xrange(1,self.nHiddenLayers)):
            dZ["Z" + str(layer)] = dZ["Z"+str(layer+1)].dot(self.W["W"+str(layer)].T) * self.Z["Z"+str(layer)]*(1-self.Z["Z"+str(layer)])
            dW["W" + str(layer)] = self.Z["Z" + str(layer)].T.dot(dZ["Z"+str(layer+1)])
            db["b" + str(layer)] = dZ["Z"+str(layer+1)].sum(axis=0)
        dW["W0"] = X.T.dot(dZ["Z1"])
        db["b0"] = dZ["Z1"].sum(axis=0)   
        
        for weight in self.W:
            self.W[weight] += self.learningRate * dW[weight]
        for bias in self.b:
            self.b[bias] += self.learningRate * db[bias]
            
    def train(self):
        for i in xrange(self.nBatches-1):
            Xtrain = self.X[i * self.batchSize : (i+1) * self.batchSize]
            Ttrain = self.Labels[i * self.batchSize : (i+1) * self.batchSize]
            self.backProp(Xtrain,Ttrain)  
        i += 1
        Xtrain = self.X[i*self.batchSize:]
        Ttrain = self.Labels[i*self.batchSize:]
        self.backProp(Xtrain,Ttrain)
        self.nEpochs += 1
        if self.nEpochs % self.saveInterval == 0:
            Predictions = self.forwardProp(self.X)    
            self.cost.append(self.costFunction(self.Labels,Predictions) * self.batchSize / self.nSamples)    
  
