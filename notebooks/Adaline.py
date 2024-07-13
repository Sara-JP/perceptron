import numpy as np
from sklearn.metrics import accuracy_score
class Adaline:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        # The weighted sum calculation (dot product)
        weighted_sum = np.dot(input, self.weights)
        return weighted_sum

    def activation(self, weighted_input):
        # Identity function (linear activation)
        return weighted_input

    def predict(self, inputs):
        new_inputs = np.insert(inputs, 0, 1, axis=1)  # Adding bias term
        predictions = []
        for myinput in new_inputs:
            weighted_input = self.weighting(myinput)
            prediction = self.activation(weighted_input)
            if prediction >= 0:
                predictions.append(1)
            else:
                predictions.append(-1)
        predictions = np.array(predictions)
        return predictions

    def fit(self, inputs, outputs, learning_rate=0.01, epochs=64):
        new_inputs = np.insert(inputs, 0, 1, axis=1)  # Adding bias term
        self.weights = np.random.rand(new_inputs.shape[1])  # Weight initialization
        for epoch in range(epochs):
            weighted_input = self.weighting(new_inputs)
            prediction = self.activation(weighted_input)
            diff = outputs - prediction  # Calculate the error
            self.weights += learning_rate * np.dot(diff,new_inputs)  # Update the weights
            predictions = self.predict(inputs)
            print(f'Epoch {epoch+1}/{epochs}, Accuracy: {accuracy_score(predictions, outputs):.4f}')

