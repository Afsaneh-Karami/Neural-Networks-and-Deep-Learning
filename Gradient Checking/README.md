## Implement gradient checking to verify the accuracy of your backprop implementation <br />

 In a deep learning model, you know that backpropagation is quite challenging to implement, and sometimes has bugs. So, you calculate gradient checking to be really certain that your implementation of backpropagation is correct. Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient. 
 
 The order of functions to check the accuracy of backward propagation: <br /> (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/main%20program)):

1) Begin by importing required packages [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/main%20program)) (line:2):<br /><br />
2) Load the dataset (GOTO [datasets link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/datasets)) 
   * X, Y, parameters = datasets()
3) Using forward_propagation_n function to calculate cost, cache (GOTO [forward_propagation_n link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/forward_propagation_n) <br />
   * cost, cache = forward_propagation_n(X, Y, parameters) <br /><br />
4) Using backward_propagation_n fuction to calculate grad. <br />(GOTO [backward_propagation_n link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/backward_propagation_n) )
   * grad = backward_propagation_n(X, Y, cache)<br />
   
 <img width="658" alt="NDgrad_kiank" src="https://user-images.githubusercontent.com/78735911/137982858-a8f36893-9793-4d78-849e-2703d65c6c44.png">

5) Using gradient_check_n function to calculate gradapprox and difference between grad and gradapprox (GOTO [gradient_check_n link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Gradient%20Checking/gradient_check_n) )
  
   The numerical approximation of the gradient or gradapprox calculate based on the following equation:<br />
   ![Capture3](https://user-images.githubusercontent.com/78735911/137984527-843b313f-f1e9-4d58-9270-e8451cc592e4.JPG)
   The difference between grad and gradapprox calculate based on the following equation:<br />
   ![Capture1](https://user-images.githubusercontent.com/78735911/137983064-e2d0ce05-c400-4429-b09e-80835e91129a.JPG)
   In gradient_check_n function I used three other fuctions, dictionary_to_vector, vector_to_dictionary, and gradients_to_vector. It converts the dictionary into a vector or the vector to dictionary. As you see in this picture.
   * theta, keys = dictionary_to_vector(parameters)<br />
   * The function "dictionary_to_vector()" converts the "parameters" dictionary into a vector called "values", obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them. <br />
   * theta = gradients_to_vector(gradients)<br />
   * parameters = vector_to_dictionary(theta)<br />
   <img width="627" alt="dictionary_to_vector" src="https://user-images.githubusercontent.com/78735911/137982751-57284551-c1e7-4940-b6e8-bbe323c44443.png">
