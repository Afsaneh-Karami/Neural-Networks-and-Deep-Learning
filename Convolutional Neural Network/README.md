## Applying convolutional and pooling layer to neural network<br />

In this file, I implemented convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and backward propagation. File "test_signs.h5" and "train_signs.h5" were used as datasets. I wrote my code in Jupyter notebook.<br />
The order of functions to make CONV and POOL filters:<br />
1. A convolution layer transforms an input volume into an output volume of different size. <br />
1.1 Zero-padding adds zeros around the border of an image. The padding is applied to the height and width of an image, as illustrated in below.(GOTO [zero_pad link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/zero_pad)) 
<img width="574" alt="PAD" src="https://user-images.githubusercontent.com/78735911/144033668-5a135959-337a-4edd-b3cf-6f0d7e25e299.png">
 The main benefits of padding are: <br />
a. It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the 
  height/width would shrink as you go to deeper layers.<br />   
b. It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.<br /> 
 X is a python numpy array of shape (m, n_H, n_W, n_C) so padding should influence n_H and n_W:<br />  
 X_pad =np.pad(X,((0,0),( pad,pad),( pad, pad),(0,0)),mode='constant', constant_values = (0,0))<br />
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
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. The two types of pooling layers are:
* Max-pooling layer: slides an (f, ff,f) window over the input and stores the max value of the window in the output.
* Average-pooling layer: slides an (f, ff,f) window over the input and stores the average value of the window in the output.
they have hyperparameters such as the window size f

