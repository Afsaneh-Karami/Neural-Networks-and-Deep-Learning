## Defining an L-layer Neural Network with L2 Regularization and Dropout Regularization <br />

In this file, I used Neural Network (NN) with two hidden layers to classify blue and red dots in the following picture. In order to avoid overfitting, I applied two regularization method to overcome mentioned problem including:
 1. L2 regularization (GOTO Folder [L2 Regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Regularization/L2%20Regularization))<br />
2. Dropout Regularization (GOTO Folder [Dropout Regularization link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Regularization/Dropout%20Regularization))<br /><br />


  *  File "data.mat" was used as traine and test datasets. I used train datasets to choose the parameters W and b of the NN model. The model was applied to train and test datasets to classify colorful dots. File "data.mat" is in Datasets folder.  <br />
   * Folder Initialization including three function ("zeros","random" or "he"):<br />
   initialize_parameters_zeros function  initializes the weights to zero.<br />
   initialize_parameters_random function  initializes the weights to random value.<br />
   initialize_parameters_he function initializes the weights to random values scaled according to "Xavier initialization". This function multiply random parameters by sqrt(2./layers_dims[l-1]). <br />
   
