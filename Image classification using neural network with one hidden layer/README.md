## Defining an L-layer Neural Network to classify cat images<br />

In this file I used Neural Network with one hidden layer to classify cat images. Traing set was used to found the parameters W1,W2,b1, and b2 and applied these parameters to predict test set images. I wrote my code in Jupyter notebook. The order of functions to make a nueral network model for classification:

1) Begin by importing required packages (GOTO [package link]() )
2) Load the dataset : 
   * train_x, train_y, test_x, test_y, classes = load_data()
4) Define the layers, where layers_dim is an array: 
   * layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
6) Train the model with the function L_layer_model(): 
   * parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True) (GOTO [a link](https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/L-layer-learning) )
   * This function first initializes the parameters W (weights) and b (bias) to random values.
  
   * For num_iterations, loops through and computes the gradient decent for the cost function:
    * makes a forward propagation (using Relu for layers 1 to L-1, and sigmoid for L), computes the cost, makes a backward propagation:
     * AL, caches = L_model_forward(X, parameters) (GOTO [a link](https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/L-model-design))
     * cost=compute_cost(AL, Y) (GOTO [a link]( https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/cost-function))
     * grads= L_model_backward(AL, Y, caches) (GOTO [a link](https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/L-model-backward-propagation))
     * parameters=update_parameters(parameters, grads, learning_rate) (GOTO [a link](https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/update-parameters))
  

    
5) To compute the accuracy:
    * pred_train = predict(train_x, train_y, parameters)
    * pred_test = predict(test_x, test_y, parameters)
  
