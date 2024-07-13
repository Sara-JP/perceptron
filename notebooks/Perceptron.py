import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron:
    def __init__(self):
        self.weights = None

    def weighting(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return weighted_sum # To-Do

    def activation(self, weighted_input):
        if weighted_input>=0:
            return 1
        else:
            return -1

    def predict(self, inputs):
            
        new_inputs = np.insert(inputs, 0, 1, axis=1)
        # a list of final prediction for each test sample
        predictions = []
        for myinput in new_inputs:
            weighted_input = self.weighting(myinput)
            prediction = self.activation(weighted_input)
            predictions.append(prediction)
        # converting the list to a numpy array
        predictions = np.array(predictions)
        return predictions

    def fit(self, inputs, outputs, learning_rate, epochs):
        
        new_inputs = np.insert(inputs, 0, 1, axis=1)
        # initializing the weights
        self.weights = np.random.rand(new_inputs.shape[1])
        # training loop
        for epoch in range(epochs):
            for sample, target in zip(new_inputs, outputs):
                weighted_input = self.weighting(sample)
                prediction = self.activation(weighted_input)
                diff =  target-prediction
                self.weights += learning_rate * diff * sample
            predictions = self.predict(inputs)
            print(f'Epoch {epoch+1}/{epochs}, Accuracy: {accuracy_score(predictions, outputs):.4f}')

