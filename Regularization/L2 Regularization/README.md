## Defining an L-layer Neural Network with L2 Regularization <br />

In this file, I used Neural Network (NN) with two hidden layers to classify blue and red dots in the following picture. In order to avoid overfitting, I applied L2 regularization.  File "data.mat" was used as training and test datasets. I used train datasets to choose the parameters W and b of the NN model. The model was applied to train and test datasets to classify colorful dots. File "data.mat" is in Datasets folder. I wrote my code in Jupyter notebook.<br />
Note : With L2 Regularization what we do is penelize the weight matrices from being too large. In this way reducing  the probability of overfitting.<br />
The order of functions to make an L-Layer NN model with L2 Regularization for classification <br /> (GOTO [prediction_L2Regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/prediction_L2Regularization)):

1) Begin by importing required packages (GOTO [package link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/import%20package))<br /><br />
2) Load the trainig dataset (GOTO [load_dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Loading%20the%20Dataset)) 
   * train_x, train_y, test_x, test_y = load_2D_dataset()
   * file: "data.mat" was used for the train set and test set respectively. you can find it in the Datasets folder. 
3) Define the size of input, hidden, and output layers (GOTO [layer_sizes link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/model-with_L2Regularization)) (line: 21 )
   * layers_dims = [X.shape[0], 20, 3, 1] <br /><br />
4) Running the model-with_L2Regularization fuction which does all forward and backward propagation and gives parameters. <br />(GOTO [model-with_L2Regularization](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/model-with_L2Regularization) )
   * parameters = model-with_L2Regularization(X, Y, learning_rate, num_iterations, print_cost = True, lambd, initialization)<br />
   * Calculation steps of model-with_L2Regularization function including: <br /><br />
            4.1. Initialize parameters W and b with one of three functions ("zeros","random" or "he") for all layers (GOTO [Initialization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Regularization/Initialization) )<br />
                parameters = initialize_parameters_he(layers_dims)  
                   Note: He Initialization is "Xavier initialization" that multiply random parameters by sqrt(2./layers_dims[l-1]) <br /> 
                parameters = initialize_parameters_random(layers_dims)<br /> 
                parameters = initialize_parameters_zeros(layers_dims)<br /> 
                   Note:Three functions are in initialization folder (GOTO [Initialization](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Regularization/Initialization))<br /> <br /> 
           4.2. Using forward_propagation_wirh_L2 regularization function to calculate Z and A based on the trainig sets <br />(GOTO [forward_propagation_wirh_L2 regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/forward_propagation_wirh_L2%20regularization))<br />
                A3, caches = forward_propagation_with_L2 regularization (X, parameters) <br />
                Activation function for layer 1 and 2 is relu, and for the last layer the sigmoid fuction is used.<br /><br /> 
           4.3. Using compute_cost_with_L2regularization fuction to calculate cost <br />(GOTO [compute_cost_with_L2regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/compute_cost_with_L2regularization ))<br />
                cost = compute_cost_with_L2regularization(A3, Y, parameters, lambd) <br /><br />
           4.4. Using backward_propagation_with_L2regularization function to calculate dA, dW, and db <br />(GOTO [backward_propagation_with_L2regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/Backward_propagation_with_L2regularization))<br />
                gradients  = backward_propagation_with_L2regularization(X, Y, cache, lambd)<br /><br />
           4.5. Using update_parameters fuction to update parameters W and b with the specified learning_rate <br />(GOTO [update_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/update_parameters))<br />
                parameters = update_parameters (parameters, gradients , learning_rate)<br /><br />
5) Using predict-with-L2Regularization fuction to estimate the accuracy of neural network model for datasets <br /> (GOTO [predict-with-L2Regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/L2%20Regularization/predict-with-L2Regularization))<br />
   * p = predict-with-L2Regularization(X, y, parameters) <br />
      
