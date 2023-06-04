# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is ran, it executes the steps indicated in the assignment.
import numpy as np

class Perceptron:
    
    def __init__(self, act_function, weights, bias):
        self.act_function = act_function
        self.weights = np.asarray(weights)
        self.inputs = np.asarray(weights)
        self.bias = bias
        self.result = 0
        
    def calc_z(self):
        result = np.dot(self.inputs, self.weights) + self.bias
        self.result = result
        return result
    
    def calc_result(self):
        result = self.act_function(self.calc_z())
        self.result = result
        return result
    
        
        

class ANN:
    #this is hardcoded to work with exactly 3 layers: an input layer, a hidden layer, and an output layer.
    
    def __init__(self, hidden_layer, output_layer, loss_function, output_activation, hidden_layer_der):     
        self.hidden_layer = np.asarray(hidden_layer)
        self.output_layer = np.asarray(output_layer)
        self.loss_function = loss_function #the loss function should be a function that takes two parameters: a list of expected, and a list of actual values
        self.output_activation = output_activation
        self.hidden_layer_der = hidden_layer_der #this probably should not be given here
       
    def feedforward(self, input_layer):
        hidden_layer = self.hidden_layer
        output_layer = self.output_layer
        
        hidden_results = np.empty(len(hidden_layer))
        #we iterate over every layer
        for i in range(len(hidden_layer)):
                curr_node = hidden_layer[int(i)]
                curr_node.inputs = input_layer
                print(curr_node.calc_result())
                hidden_results[i] = curr_node.calc_result()
                
        output_results = np.empty(len(output_layer))
        for i in range(len(output_layer)):
                curr_node = output_layer[i]
                curr_node.inputs = hidden_results
                output_results[i] = curr_node.calc_z()
        output_results = self.output_activation(output_results)
        for i in range(len(output_layer)):
            output_layer[i].result = output_results[i]
        return output_results
            
    #this method should give us the deltaws, and the deltabiases
    #biases are not even added we can add them later
    #use_relu input added later with a default to false, as to not break our previously written code
    def backpropagation(self, input_layer, expected_result, use_relu=False):
        results = self.feedforward(input_layer)
        hidden_layer = self.hidden_layer
        output_layer = self.output_layer
        
        loss_value = self.loss_function(expected_result, results)
        deltaWinputtohidden = np.empty((len(hidden_layer), len(input_layer)))
        deltaWhiddentooutput = np.empty((len(output_layer), len(hidden_layer)))
        deltaBiasHidden = np.empty(len(hidden_layer))
        deltaBiasOutput = np.empty(len(output_layer))
        
        #first, we do the output layer. Hardcoded to work for softmax function :)
        expAll = 0
        expAll = np.sum(np.exp(np.vectorize(lambda x: x.result)(output_layer)))
            
        derivatives = (np.exp(np.vectorize(lambda x: x.result)(output_layer))*expAll - np.exp(2 * np.vectorize(lambda x: x.result)(output_layer)) ) / ( expAll * expAll)
        deltaBiasOutput = -np.vectorize(lambda x: x.result)(output_layer)
        deltaBiasOutput[expected_result - 1] += 1
        deltaWhiddentooutput = (np.tile(np.vectorize(lambda x: x.result)(hidden_layer), (7,1)).T * derivatives * deltaBiasOutput).T
        
        for l in range(len(hidden_layer)):
            sum_error = np.sum(np.vectorize(lambda x:x.weights[l])(output_layer) * derivatives * deltaBiasOutput)
            deltaBiasHidden[l] = sum_error
            
            
        j = hidden_layer[l]
        #sigmoid derivative hardcoded
        if use_relu:
             deltaWinputtohidden = (np.tile(input_layer, (len(hidden_layer),1)).T * np.vectorize(lambda j: 0 if (np.isclose(0, j.result)) else 1)(hidden_layer) * deltaBiasHidden).T
        else:
            deltaWinputtohidden = (np.tile(input_layer, (len(hidden_layer),1)).T * np.vectorize(lambda j: j.result *(1 - j.result))(hidden_layer) * deltaBiasHidden).T
                
        return deltaWinputtohidden, deltaWhiddentooutput, deltaBiasHidden, deltaBiasOutput
    
    #given delta vectors, it updates the nodes
    def update(self, deltaWinputtohidden, deltaWhiddentooutput, deltaBiasHidden, deltaBiasOutput, alpha):
        hidden_layer = self.hidden_layer
        output_layer = self.output_layer
        
        for i in range(deltaWinputtohidden.shape[0]):
            hidden_layer[i].bias = hidden_layer[i].bias + alpha * deltaBiasHidden[i]
            hidden_layer[i].weights = np.add(hidden_layer[i].weights, np.multiply(alpha, deltaWinputtohidden[i]))
            #for j in range(deltaWinputtohidden.shape[1]):
            #    self.hidden_layer[i].weights[j] = self.hidden_layer[i].weights[j] + alpha * deltaWinputtohidden[i][j]
        for i in range(deltaWhiddentooutput.shape[0]):
            output_layer[i].bias = output_layer[i].bias + alpha * deltaBiasOutput[i]
            output_layer[i].weights = np.add(output_layer[i].weights, np.multiply(alpha, deltaWhiddentooutput[i]))
            #for j in range(deltaWhiddentooutput.shape[1]):
            #    self.output_layer[i].weights[j] = self.output_layer[i].weights[j] + alpha * deltaWhiddentooutput[i][j]
        
                
        
        
        
        
        
            
    
    