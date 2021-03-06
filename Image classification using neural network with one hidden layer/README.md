## Defining a 3-layer Neural Network to classify cat images<br />

In this file, I used Neural Network (NN) with one hidden layer to classify cat images. The training set was used to found the parameters W1, W2, b1, and b2 of NN and then applied the model to testset to classify images as cat or non-cat. I wrote my code in Jupyter notebook.<br />
The order of functions to make a neural network model for classification :

1) Begin by importing required packages (GOTO [package link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/import%20package) )
2) Load the trainig dataset (GOTO [load_dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/load%20data) ): 
   * train_x, train_y, test_x, test_y = load_data()
   * file: train_catvnoncat.h5 and test_catvnoncat.h5 were used for the training set and test set respectively. you can find them in the Datasets folder.
3) Victorize the train_x and train_y (GOTO [vectorization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/vectorization )):
   * v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
     
4) Define the size of input and output layers (GOTO [layer_sizes link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/layer_sizes) ): 
   * n_x, n_y=layer_sizes(train_set_x_orig, train_set_y_orig))
5)Initialize parameters W1.b1,W2,b2 with arandom value(GOTO [initialize_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/initialize_parameters) )
   *parameters =initialize_parameters(n_x, n_h, n_y) # n_h is the units of hidden layer
  

6) Applying loop of Gradient descent for num_iterations to find the parameters W1, W2 ,b1 , and b2 which minimize the cost function J (GOTO [nn_model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/neural%20network%20model) (line: 12-20 ): 
   * Using forward propagation to calculate Z1, A1, Z2, and A2 (GOTO [Forward propagation link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/forward%20propagation ))<br />
     A2, cache =forward_propagation(X, parameters)
   
     
     
   * Using compute_cost fuction (GOTO [compute_cost link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/compute%20cost ))<br />
     cost =compute_cost(A2, Y)

     
   * Using backward propagation to calculate dW1, dW2, db1, and db2 (GOTO [backward propagation link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/backward%20propagation ))<br />
     grads =backward_propagation(parameters, cache, X, Y)
   * Using update_parameters fuction to update parameters W1, W2 ,b1 , and b2 with the learning_rate = 1.2 (GOTO [update_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/update_parameters ))<br />
     parameters =update_parameters(parameters, grads, learning_rate = 1.2)
7) writing Predict fuction to predict result by the neural network model (GOTO [predict link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/predict))<br />
    * predictions=predict(parameters, X)
8) Applying nn_model to test dataset to classify cat images and then compute the accuracy of it (GOTO [Apply nn_model on test set link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Image%20classification%20using%20neural%20network%20with%20one%20hidden%20layer/test%20nn_model%20on%20test%20set))<br />
     * X=test_set_x_orig<br />
     * Y=test_set_y_orig<br />
    * parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)<br />
    * predictions = predict(parameters, X)<br />
  

    

