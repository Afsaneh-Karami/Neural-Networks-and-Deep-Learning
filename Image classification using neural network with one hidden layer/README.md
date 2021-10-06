######Defining an Three-layer Neural Network to classify cat images

Usinng Jupyter notebook. The order of functions to make a nueral network model for classification:

Begin by importing required packages (GOTO a link )

Load the dataset :

train_x, train_y, test_x, test_y, classes = load_data()
Define the layers, where layers_dim is an array:

layers_dims = [12288, 20, 7, 5, 1] # 4-layer model
Train the model with the function L_layer_model():

parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True) (GOTO a link )

This function first initializes the parameters W (weights) and b (bias) to random values.

For num_iterations, loops through and computes the gradient decent for the cost function:

makes a forward propagation (using Relu for layers 1 to L-1, and sigmoid for L), computes the cost, makes a backward propagation:

AL, caches = L_model_forward(X, parameters) (GOTO a link)

cost=compute_cost(AL, Y) (GOTO a link)

grads= L_model_backward(AL, Y, caches) (GOTO a link)

parameters=update_parameters(parameters, grads, learning_rate) (GOTO a link)

To compute the accuracy:

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
