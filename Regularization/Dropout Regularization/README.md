## Defining an L-layer Neural Network with Dropout Regularization <br />

In this file, I used Neural Network (NN) with two hidden layers to classify blue and red dots in the following picture. In order to avoid overfitting, I applied dropout regularization.  File "data.mat" was used as training and test datasets. I used train datasets to choose the parameters W and b of the NN model. The model was applied to train and test datasets to classify colorful dots. File "data.mat" is in Datasets folder. I wrote my code in Jupyter notebook.<br />
Note 1: With dropout what we do is go through each of layers of the network and set some probability of eliminating a node in network. in this way we demolished network and have much smaller NN. So, reducing  the probability of overfitting.<br />
Note 2: The keep_prob for input layer is 1, because we do not want to lose the input information.<br />
Note 3: Not use drop out at test time because we do not want our output to be random (keep_prob=1).<br />
![Capture](https://user-images.githubusercontent.com/78735911/137906920-87d2585c-ca4a-47bf-91e4-2f8c7832d352.JPG)

The order of functions to make an L-Layer NN model with Dropout Regularization for classification <br /> (GOTO [prediction_DropoutRegularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/prediction_DropoutRegularization)):

1) Begin by importing required packages (GOTO [package link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/import%20package))<br /><br />
2) Load the trainig dataset (GOTO [load_dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Loading%20the%20Dataset)) 
   * train_x, train_y, test_x, test_y = load_2D_dataset()
   * file: "data.mat" was used for the train set and test set respectively. you can find it in the Datasets folder. 
3) Define the size of input, hidden, and output layers (GOTO [layer_sizes link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/model_with_DropoutRegularization) (line: 21 )
   * layers_dims = [X.shape[0], 20, 3, 1] <br /><br />
4) Running the model-with_DropoutRegularization fuction which does all forward and backward propagation and gives parameters. (GOTO [model-with_DropoutRegularization](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/model_with_DropoutRegularization) )
   * parameters = model-with_DropoutRegularization(X, Y, learning_rate, num_iterations, print_cost = True, keep_prob)<br />
   * Calculation steps of model-with_DropoutRegularization function including: <br /><br />
            4.1. Initialize parameters W and b with one of three fuctions ("zeros","random" or "he") for all layers (GOTO [Initialization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Regularization/Initialization) )<br />
                parameters = initialize_parameters_he(layers_dims)  
                   Note: He Initialization is "Xavier initialization" that multiply random parameters by sqrt(2./layers_dims[l-1]) <br /> 
                parameters = initialize_parameters_random(layers_dims)<br /> 
                parameters = initialize_parameters_zeros(layers_dims)<br /> 
                   Note:Three fuctions are in initialization folder (GOTO [Initialization](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Regularization/Initialization))<br /> <br /> 
           4.2. Using forward_propagation_with_dropout to calculate Z and A based on the trainig sets <br />(GOTO [forward_propagation_with_dropout link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/forward_propagation_with_dropout))<br />
                A3, caches = forward_propagation_with_dropout(X, parameters, keep_prob)<br />
                Activation function for layer 1 and 2 is relu, and for the last layer the sigmoid fuction is used.<br /><br /> 
           4.3. Using compute_cost_with_DropoutRegularization fuction to calculate cost <br />(GOTO [compute_cost_with_DropoutRegularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/compute_cost_with_DropoutRegularization ))<br />
                cost = compute_cost_with_DropoutRegularization(A3, Y, parameters) <br /><br />
           4.4. Using backward_propagation_with_dropout to calculate dA, dW, and db (GOTO [backward_propagation_with_dropout link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/Backward_propagation_with_dropout))<br />
                gradients  = backward_propagation_with_dropout(X, Y, cache, keep_prob)<br /><br />
           4.5. Using update_parameters fuction to update parameters W and b with the specified learning_rate (GOTO [update_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/update_parameters))<br />
                parameters = update_parameters(parameters, gradients , learning_rate)<br /><br />
  5) Using predict-with-DropoutRegularization fuction to estimate the accuracy of neural network model for datasets (GOTO [Predict-with-Dropour Regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/Predict-with-Dropour%20Regularization)<br />
    * predict-with-DropoutRegularization(X, y, parameters) <br />
      
  
