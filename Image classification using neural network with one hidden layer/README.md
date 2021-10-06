## Defining an 3-layer Neural Network to classify cat images<br />

In this file I used Neural Network with one hidden layer to classify cat images. Traing set was used to found the parameters W1,W2,b1, and b2 and applied these parameters to predict test set images. I wrote my code in Jupyter notebook. The order of functions to make a nueral network model for classification:

1) Begin by importing required packages (GOTO [package link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/import%20package) )
2) Load the trainig dataset (GOTO [load_dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/load%20data) ): 
   * train_x, train_y, test_x, test_y = load_data()
3) Define the size of input and output layers (GOTO [load_dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/layer_sizes) ): 
   * n_x, n_y=layer_sizes(train_set_x_orig, train_set_y_orig)
4)Initialize parameters W1.b1,W2,b2 with arandom value(GOTO [initialize_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/initialize_parameters) )
   *parameters =initialize_parameters(n_x, n_h, n_y) # n_h is the units of hidden layer
4) Applying Loop of Gradient descent to find the parameters W1,W2,b1, and b2 which minimize the cost function J (GOTO [nn_model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/neural%20network%20model) (line: 12-20 ): 
   * Using Forward propagation to calculate Z1,A1,Z2, and A2 (GOTO [Forward propagation](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/forward%20propagation )
   * Using compute_cost fuction (GOTO [compute_cost](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/compute%20cost )
   * Using backward propagation to calculate dW1,dW2,db1, and db2 (GOTO [backward propagation](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/backward%20propagation )
  
   * For num_iterations, loops through and computes the gradient decent for the cost function:
    * makes a forward propagation (using Relu for layers 1 to L-1, and sigmoid for L), computes the cost, makes a backward propagation:
     * AL, caches = L_model_forward(X, parameters) (GOTO [a link](https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/L-model-design))
     * cost=compute_cost(AL, Y) (GOTO [a link]( https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/cost-function))
     * grads= L_model_backward(AL, Y, caches) (GOTO [a link](https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/L-model-backward-propagation))
     * parameters=update_parameters(parameters, grads, learning_rate) (GOTO [a link](https://github.com/Farzane-Ka/deep-learning/blob/main/nn-image-classification/update-parameters))
  

    
5) To compute the accuracy:
    * pred_train = predict(train_x, train_y, parameters)
    * pred_test = predict(test_x, test_y, parameters)
  
