## Implement gradient checking to verify the accuracy of your backprop implementation <br />

 In a deep learning model, you know that backpropagation is quite challenging to implement, and sometimes has bugs. So, you calculate gradient checking to be really certain that your implementation of backpropagation is correct. Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient. 
 
 The order of functions to check the accuracy of backward propagation (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/main%20program)):

1) Begin by importing required packages [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/main%20program)) (line:2):<br /><br />
2) Load the dataset (GOTO [datasets link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/datasets)) 
   * X, Y, parameters = datasets()<br /><br />
3) Using forward_propagation_n function to calculate cost, cache (GOTO [forward_propagation_n link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/forward_propagation_n)): <br />
   * cost, cache = forward_propagation_n(X, Y, parameters) <br /><br />
4) Using backward_propagation_n fuction to calculate grad (GOTO [backward_propagation_n link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/backward_propagation_n) ):
   * grad = backward_propagation_n(X, Y, cache)<br />
   

5) Using gradient_check_n function to calculate gradapprox and difference between grad and gradapprox <br />(GOTO [gradient_check_n link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/gradient_check_n) )
  
    * The numerical approximation of the gradient or gradapprox calculate based on the following equation:<br />
   ![Capture3](https://user-images.githubusercontent.com/78735911/137984527-843b313f-f1e9-4d58-9270-e8451cc592e4.JPG)<br />
    * The difference between grad and gradapprox calculate based on the following equation:<br /><br />
   ![Capture1](https://user-images.githubusercontent.com/78735911/137983064-e2d0ce05-c400-4429-b09e-80835e91129a.JPG)<br />
    * In gradient_check_n function I used three other fuctions, dictionary_to_vector, vector_to_dictionary, and gradients_to_vector. It converts the dictionary into a vector or the vector to dictionary. 
     * theta, keys = dictionary_to_vector(parameters) <br />
    * The function "dictionary_to_vector()" converts the parameters dictionary into a vector called values, obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them. <br />
    * theta = gradients_to_vector(gradients) <br /> 
    * parameters = vector_to_dictionary(theta) <br />
  
