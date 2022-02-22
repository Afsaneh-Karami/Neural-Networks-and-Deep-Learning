## Applying convolutional and pooling layer to neural network<br />

In this file, I implemented convolutional (CONV) and pooling (POOL) layers in NumPy, including both forward propagation and backward propagation. File "test_signs.h5" and "train_signs.h5" were used as datasets. I wrote my code in Jupyter notebook.<br />
The order of functions to make CONV and POOL filters:<br />
 A convolution layer transforms an input volume into an output volume of different sizes.<br /><br />
1. zero-padding adds zeros around the border of an image. The padding is applied to the height and width of an image, as illustrated below. (GOTO [zero_pad link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/zero_pad)) 
<img width="574" alt="PAD" src="https://user-images.githubusercontent.com/78735911/144033668-5a135959-337a-4edd-b3cf-6f0d7e25e299.png">
 The main benefits of padding are: <br />
a. It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks since otherwise the 
  height/width would shrink as you go to deeper layers.<br />   
b. It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.<br /> 
 X is a python NumPy array of shapes (m, n_H, n_W, n_C) so padding should influence n_H and n_W:<br />  
 X_pad =np.pad(X,((0,0),( pad,pad),( pad, pad),(0,0)),mode='constant', constant_values = (0,0))<br /><br />
2. Single Step of Convolution <br />
 Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
 of the previous layer. 
 
![Convolution_schematic](https://user-images.githubusercontent.com/78735911/144194485-20728af7-df66-499c-9b6e-2c59582370d0.gif)<br />
(GOTO [Single Step of Convolution link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/Single%20Step%20of%20Convolution))<br /><br />
3. Convolutional Neural Networks - Forward Pass (GOTO [Convolutional Neural Networks - Forward Pass link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/Convolutional%20Neural%20Networks%20-%20Forward%20Pass)) <br />
In the forward pass, you will take many filters and convolve them on the input. Each 'convolution' gives you a 2D matrix output. You will then stack these outputs to get a 3D volume.
The formulas relating the output shape of the convolution to the input shape are:
![Capture](https://user-images.githubusercontent.com/78735911/144050854-c79b2fc0-f16c-4802-a8a5-de9b86e9a9be.PNG)<br />
Note:<br />
To define a_slice you need to  define its corners vert_start, vert_end, horiz_start and horiz_end.<br />
vert_start = h*stride <br />
vert_end =vert_start+f <br />
horiz_start =w*stride <br />
horiz_end =horiz_start+f <br />
<img width="508" alt="vert_horiz_kiank" src="https://user-images.githubusercontent.com/78735911/144199090-c601b9e3-c2ef-47be-a6ee-6caa3d19477c.png">

4. Pooling Layer (GOTO [Forward Pooling link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/Forward%20Pooling)) <br />
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to their position in the input. The two types of pooling layers are: <br />
* Max-pooling layer: slides an (f,f) window over the input and stores the max value of the window in the output. <br />
 A[i, h, w, c] =np.max(a_prev_slice)
* Average-pooling layer: slides an (f,f) window over the input and stores the average value of the window in the output. <br />
A[i, h, w, c] =np.mean(a_prev_slice)<br />
They have hyperparameters such as the window size f. <br />
5. Backpropagation in Convolutional Neural Networks (GOTO [conv_backward link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/conv_backward)) <br />
Implement the conv_backward function, You should compute the derivatives dA, dW,and db.Then sum over all the training examples, filters, heights, and widths. <br />
5.1 Computing dA:<br />
This is the formula for computing  dA  with respect to the cost for a certain filter  Wc  and a given training example:<br />
![1](https://user-images.githubusercontent.com/78735911/154953443-c1965763-07e2-4151-9230-f1a4920ee1de.PNG) <br />
Where  Wc  is a filter and  dZhw  is a scalar corresponding to the gradient of the cost with respect to the output of the conv layer Z at the hth row and wth column (corresponding to the dot product taken at the ith stride left and jth stride down). Note that at each time, you multiply the the same filter  Wc  by a different dZ when updating dA. We do so mainly because when computing the forward propagation, each filter is dotted and summed by a different a_slice. Therefore when computing the backprop for dA, you are just adding the gradients of all the a_slices. In code, inside the appropriate for-loops, this formula translates into:<br />
da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c] <br /><br />
5.2 Computing dW:<br />
This is is the formula for computing  dWc  ( dWc  is the derivative of one filter) with respect to the loss:<br />
![22](https://user-images.githubusercontent.com/78735911/154954538-8520a638-4d52-438e-a05c-d1ca7cb0b8f9.PNG)<br />
Where  aslice  corresponds to the slice which was used to generate the activation  Zij . Hence, this ends up giving us the gradient for  W  with respect to that slice. Since it is the same  W , we will just add up all such gradients to get dW .In code, inside the appropriate for-loops, this formula translates into:<br />
dW[:,:,:,c] += a_slice * dZ[i, h, w, c] <br /><br />
5.3 Computing db:<br />
This is the formula for computing  db  with respect to the cost for a certain filter  Wc:<br />
![3](https://user-images.githubusercontent.com/78735911/154954834-733f9d1d-c66d-4b30-8ac2-17e630563c35.PNG)<br />
As you have previously seen in basic neural networks, db is computed by summing  dZ . In this case, you are just summing over all the gradients of the conv output (Z) with respect to the cost. In code, inside the appropriate for-loops, this formula translates into:<br />
db[:,:,:,c] += dZ[i, h, w, c] <br /><br />
6. Backward propagation on a pooling layer(GOTO [pool_backward link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/pool_backward))<br />
Even though a pooling layer has no parameters for backprop to update, you still need to backpropagate the gradient through the pooling layer in order to compute gradients for layers that came before the pooling layer.<br />
Implement the pool_backward function in both modes ("max" and "average"). You will once again use 4 for-loops (iterating over training examples, height, width, and channels). You should use an if/elif statement to see if the mode is equal to 'max' or 'average'. If it is equal to 'average' you should use the distribute_value() function  to create a matrix of the same shape as a_slice by a mutiplier. Otherwise, the mode is equal to 'max', and you will create a mask with create_mask_from_window() and multiply it by the corresponding value of dA.<br />
6.1 mask for Max pooling (GOTO [create_mask_from_window link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/create_mask_from_window))<br />
creating a function that creates a "mask" matrix which keeps track of where the maximum of the matrix is. True (1) indicates the position of the maximum in X, the other entries are False (0). <br />
mask =(x==np.max(x)) <br />
6.2 distribute fuction for average pooling (GOTO [distribute_value link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/distribute_value))<br />
In max pooling, for each input window, all the "influence" on the output came from a single input value--the max. In average pooling, every element of the input window has equal influence on the output. So to implement backprop, you will now implement a helper function that reflects this. This implies that each position in the  𝑑𝑍  matrix contributes equally to output because in the forward pass, we took an average.<br />





