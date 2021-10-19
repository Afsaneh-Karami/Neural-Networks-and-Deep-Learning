## Defining an L-layer Neural Network with Dropout Regularization <br />

In this file, I used Neural Network (NN) with three hidden layers to classify blue and red doth in the following picture. In order to avoid overfitting, I applied Dropout Regularization.  File data.mat was used as training and test datasets. I used training datasets to choose the parameters W and b of the NN model. The model was applied to train and test datasets to classify colorful dots. File data.mat is in Datasets folder. I wrote my code in Jupyter notebook.<br />
Note 1: With dropout what we do is go through each of layers of the network and set some probability of eliminating a node in network. in this way we demolished network and have much smaller NN.<br />
Note 2: The keep_prob for input layer is 1, because we do not want to lose the input information.<br />
Note 3: Not use drop out at test time because we do not want our output to be random (keep_prob=1).<br />
![Capture](https://user-images.githubusercontent.com/78735911/137906920-87d2585c-ca4a-47bf-91e4-2f8c7832d352.JPG)

The order of functions to make an L-Layer NN model with Dropout Regularization for classification :

1) Begin by importing required packages (GOTO [package link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/import%20package))<br /><br />
2) Load the trainig dataset (GOTO [load_dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Loading%20the%20Dataset)) 
   * train_x, train_y, test_x, test_y = load_2D_dataset()
   * file: data.mat was used for the training set and test set respectively. you can find it in the Datasets folder. 
3) Define the size of input, hidden, and output layers (GOTO [layer_sizes link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/model_with_DropoutRegularization)(line: 21 )
   * layers_dims = [X.shape[0], 20, 3, 1] <br /><br />
4) Running the model-with_DropoutRegularization fuction which does all forward and backward propagation and gives parameters. (GOTO [model-with_DropoutRegularization](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Regularization/Dropout%20Regularization/model_with_DropoutRegularization) )
   * parameters = model-with_DropoutRegularization(X, Y, learning_rate, num_iterations, print_cost = True, keep_prob)<br />
   * Calculation steps of model-with_DropoutRegularization function including: <br /><br />
            4.1. Initialize parameters W and b with a random value for all layers (GOTO [initialize_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/A%20deep%20neural%20network%20with%20L%20layer/initialize_parameters) )<br />
                parameters = initialize_parameters (layers_dims)<br />     
           4.2. Using forward propagation to calculate Z and A based on the trainig sets (GOTO [linear_activation_forward (for L layer) link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/A%20deep%20neural%20network%20with%20L%20layer/linear_activation_forward%20(for%20L%20layer))<br />
                AL, caches = L_model_forward (X, parameters)<br />
                Activation function for layer 1,2, and 3 is relu, and for the last layer the sigmoid fuction is used.<br /><br /> 
           4.3. Using compute_cost fuction (GOTO [Cost Function link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/A%20deep%20neural%20network%20with%20L%20layer/Cost%20Function ))<br />
                cost = compute_cost (AL, Y)<br /><br />
           4.4. Using backward propagation to calculate dA, dW, and db (GOTO [linear_activation_backward (for L layer) link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/A%20deep%20neural%20network%20with%20L%20layer/linear_activation_backward%20(for%20L%20layer)))<br />
                grads = L_model_backward (AL, Y, caches)<br /><br />
           4.5. Using update_parameters fuction to update parameters W and b with the specified learning_rate (GOTO [update_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/A%20deep%20neural%20network%20with%20L%20layer/update_parameters))<br />
                parameters = update_parameters (parameters, grads, learning_rate)<br /><br />
  5) Using Predict fuction to estimate the accuracy of neural network model for datasets (GOTO [predict and accuracy link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/A%20deep%20neural%20network%20with%20L%20layer/predict%20and%20accuracy))<br />
    * predictions_train = predict (train_x, train_y, parameters)<br />
    * predictions_test = predict (test_x, test_y, parameters) <br />  
  
