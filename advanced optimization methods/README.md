## Applying optimization methods such as Gradient Descent, Momentum, RMSProp, Adam and Minibatches to speed up learning process<br />

In this file, I used some optimization methods such as gradient descent, momentum, RMSprop, adam and minibatches to speed up learning process, which can even get you to a better final value for the cost function. 
* Minibatches Gradient Descent (GOTO Folder [Mini-batch link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/advanced%20optimization%20methods/Batch%20gradient%20descent%20and%20mini-batch%20gradient%20descent))<br />
In order to build some mini-batches from the training set (X, Y), shuffling and partitioning are the two steps required to build mini-batches
1. Shuffle: Create a shuffled version of the training set (X, Y) as shown below. Each column of X and Y represents a training example. Note that the random shuffling is done synchronously between X and Y. Such that after the shuffling the  𝑖𝑡ℎ  column of X is the example corresponding to the  𝑖𝑡ℎ  label in Y. The shuffling step ensures that examples will be split randomly into different mini-batches.<br />
<img width="685" alt="1" src="https://user-images.githubusercontent.com/78735911/141608780-94e92026-a0a2-4b52-96d1-68ad01caee42.png"><br />
2. Partition: Partition the shuffled (X, Y) into mini-batches of size mini_batch_size (here 64). Note that the number of training examples is not always divisible by mini_batch_size. The last mini batch might be smaller, but you don't need to worry about this. When the final mini-batch is smaller than the full mini_batch_size, it will look like this:
<img width="560" alt="2" src="https://user-images.githubusercontent.com/78735911/141608852-3fde3c73-b712-4823-86db-31cf1c9662d6.png"><br /><br />
* Momentum (GOTO Folder [Momentum link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/advanced%20optimization%20methods/Momentum))<br />
Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations. Momentum takes into account the past gradients to smooth out the steps of gradient descent.
Implement the parameters update with momentum. The momentum update rule is, for  𝑙=1,...,𝐿 :<br />

![1](https://user-images.githubusercontent.com/78735911/141611199-0ebc7fda-70b6-41e8-ab9f-a56bb0f44f23.PNG)<br />


where L is the number of layers,  𝛽  is the momentum and  𝛼  is the learning rate. All parameters should be stored in the parameters dictionary. So, you have to tune a momentum hyperparameter  𝛽  and a learning rate  𝛼.

* Adam <br />
Adam is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp (described in lecture) and Momentum.
How does Adam work?
It calculates an exponentially weighted average of past gradients, and stores it in variables  𝑣  (before bias correction) and  𝑣𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑  (with bias correction).
It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables  𝑠  (before bias correction) and  𝑠𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑  (with bias correction).
It updates parameters in a direction based on combining information from "1" and "2".
The update rule is, for  𝑙=1,...,𝐿 :<br />
![2](https://user-images.githubusercontent.com/78735911/141611327-39338733-1510-438a-aac6-69e14083a41e.PNG) <br />

where:
t counts the number of steps taken of Adam
L is the number of layers
𝛽1  and  𝛽2  are hyperparameters that control the two exponentially weighted averages.
𝛼  is the learning rate
𝜀  is a very small number to avoid dividing by zero
As usual, all parameters are stored in the parameters dictionary.


2. Dropout Regularization (GOTO Folder [Dropout Regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Regularization/Dropout%20Regularization))<br /><br />
![Capture](https://user-images.githubusercontent.com/78735911/137906920-87d2585c-ca4a-47bf-91e4-2f8c7832d352.JPG) <br />

  *  File "data.mat" was used as traine and test datasets. I used train datasets to choose the parameters W and b of the NN model. The model was applied to train and test datasets to classify colorful dots. File "data.mat" is in Datasets folder.  <br />
   * Folder Initialization including three function ("zeros","random" or "he"):<br />
   initialize_parameters_zeros function  initializes the weights to zero.<br />
   initialize_parameters_random function  initializes the weights to random value.<br />
   initialize_parameters_he function initializes the weights to random values scaled according to "Xavier initialization". This function multiply random parameters by sqrt(2./layers_dims[l-1]). <br />
   

