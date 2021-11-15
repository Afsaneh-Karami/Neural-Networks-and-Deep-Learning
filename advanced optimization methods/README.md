## Applying optimization methods such as Gradient Descent, Momentum, RMSProp, and Adam with Minibatches to speed up learning process<br />

In this file, I used some optimization methods such as gradient descent, momentum, RMSprop, and adam to speed up the learning process, which can even get a better final value for the cost function. I wrote my code in Jupyter notebook.
The order of functions to apply different optimizers and compare them is according to the following steps: 
1. Begin by importing required packages (GOTO Folder [Import packages link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/Import%20packages))<br /><br />
2. Load the training dataset (GOTO [load_dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/Load%20dataset)) <br />
   * train_x, train_y= load_dataset()<br /><br />
3. Define the size of input, hidden, and output layers:<br />
   * layers_dims = [train_X.shape[0], 5, 2, 1]<br /><br />
4. Running the Model function which does all forward and backward propagation and gives parameters and costs function based on the optimization algorithm. (GOTO [Model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/Model) )<br /><br />
  4.1. Initialize the parameters w and b for all layers (GOTO [initialize_parameters link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/initialize_parameters)):<br /><br />
  4.2. Initialize the optimizer (GOTO [Model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/Model) lines: 31-37), which initialize the parameters related o the optimizer algorithm such as Vdw, Vdb, Sdw, and Sdb for Adam.<br /><br /><br />
  4.3. Running random_mini_batches function to create mini-batches from datasets (GOTO [Mini-Batch Gradient Descent link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/advanced%20optimization%20methods/Mini-Batch%20Gradient%20Descent)):<br />
   * minibatches = random_mini_batches(X, Y, mini_batch_size)<br />
   * In order to build some mini-batches from the training set (X, Y), shuffling and partitioning are the two steps required to build mini-batches: <br /><br />
4.3.1 Shuffle: Create a shuffled version of the training set (X, Y) as shown below. Each column of X and Y represents a training example. Note that the random shuffling is done synchronously between X and Y. Such that after the shuffling the  𝑖𝑡ℎ  column of X is the example corresponding to the  𝑖𝑡ℎ  label in Y. The shuffling step ensures that examples will be split randomly into different mini-batches.<br />
<img width="685" alt="1" src="https://user-images.githubusercontent.com/78735911/141608780-94e92026-a0a2-4b52-96d1-68ad01caee42.png"><br /><br />
4.3.2. Partition: Partition the shuffled (X, Y) into mini-batches of size mini_batch_size (here 64). Note that the number of training examples is not always divisible by mini_batch_size. The last mini batch might be smaller, but you don't need to worry about this. When the final mini-batch is smaller than the full mini_batch_size, it will look like this:<br /><br />
<img width="560" alt="2" src="https://user-images.githubusercontent.com/78735911/141608852-3fde3c73-b712-4823-86db-31cf1c9662d6.png"><br />

  4.4. Using forward propagation to calculate Z and A for each mini-batches (GOTO [forward propagation link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/forward%20propagation))<br />
       * a3, caches = forward_propagation(minibatch_X, parameters)<br /><br />
  4.5.  Using compute_cost fuction for each mini-batches (GOTO [compute_cost link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/compute_cost ))<br />
        * cost_total += compute_cost(a3, minibatch_Y)<br /><br />
  4.6. Using backward propagation to calculate dA, dW, and db for each mini-batches (GOTO [backward_propagation link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/backward_propagation))<br /> 
       * grads = backward_propagation(minibatch_X, minibatch_Y, caches)<br /><br />
  4.7. Using update_parameters function to update parameters based on the optimizer algorithm <br /><br />
     4.7.1. For updating by gradient descent(GOTO [Gradient descent link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/advanced%20optimization%20methods/Gradient%20descent))<br />
        * parameters = update_parameters_with_gd( parameters, grads, learning_rate )<br /><br />
     4.7.2. For updating by Gradient Descent with Momentum(GOTO [Momentum link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/advanced%20optimization%20methods/Momentum))<br />
        * parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate) <br />
        * Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path    taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations. Momentum takes into account the past gradients to smooth out the steps of gradient descent.
Implement the parameters update with momentum. The momentum update rule is, for  𝑙=1,...,𝐿 :<br />
![1](https://user-images.githubusercontent.com/78735911/141611199-0ebc7fda-70b6-41e8-ab9f-a56bb0f44f23.PNG)<br />
where: <br />
L: is the number of layers<br />
𝛽: is the momentum <br />
𝛼: is the learning rate<br />
All parameters should be stored in the parameters dictionary. So, you have to tune a momentum hyperparameter  𝛽  and a learning rate  𝛼. <br /><br />
      4.7.3. For updating by Adam(GOTO [Adam link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/advanced%20optimization%20methods/Adam)) <br/>   
            * parameters, v, s, _, _ = update_parameters_with_adam ( parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon )<br/> 
            * Adam is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp and Momentum.
How does Adam work? <br/>
It calculates an exponentially weighted average of past gradients, and stores it in variables  𝑣  (before bias correction) and  𝑣𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑  (with bias correction).
It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables  𝑠  (before bias correction) and  𝑠𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑  (with bias correction).
It updates parameters in a direction based on combining information.
The update rule is, for  𝑙=1,...,𝐿 :<br />
![2](https://user-images.githubusercontent.com/78735911/141611327-39338733-1510-438a-aac6-69e14083a41e.PNG) <br />
  where:<br />
  t: counts the number of steps taken of Adam<br />
  L: is the number of layers<br />
  𝛽1:  and  𝛽2  are hyperparameters that control the two exponentially weighted averages.<br />
  𝛼:  is the learning rate<br />
  𝜀: is a very small number to avoid dividing by zero<br />
  As usual, all parameters are stored in the parameters dictionary.<br /><br />
5. Using Predict function to estimate the results (GOTO [Predict link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/advanced%20optimization%20methods/Predict))<br />
    * predictions = predict(train_X, train_Y, parameters)<br />
6. Plot decision boundary for each optimizer algorithm. <br />

