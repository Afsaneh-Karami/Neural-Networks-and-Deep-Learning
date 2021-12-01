## Applying convolutional and pooling layer to neural network<br />

In this file, I implemented convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and backward propagation. File test_signs.h5 and train_signs.h5 were used as datasets. I wrote my code in Jupyter notebook.<br />
The order of functions to make CONV and POOL filters:
 
* Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
as illustrated in below.
<img width="574" alt="PAD" src="https://user-images.githubusercontent.com/78735911/144033668-5a135959-337a-4edd-b3cf-6f0d7e25e299.png">
The main benefits of padding are:
1. It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers.
2. It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.

 * Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
 of the previous layer.   
![Convolution_schematic](https://user-images.githubusercontent.com/78735911/144041948-845be123-cbf2-46cd-ae65-cc1a4c3654a4.gif)

3.3 - Convolutional Neural Networks - Forward Pass
In the forward pass, you will take many filters and convolve them on the input. Each 'convolution' gives you a 2D matrix output. You will then stack these outputs to get a 3D volume.
The formulas relating the output shape of the convolution to the input shape are:
![Capture](https://user-images.githubusercontent.com/78735911/144050854-c79b2fc0-f16c-4802-a8a5-de9b86e9a9be.PNG)

4 - Pooling Layer
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. The two types of pooling layers are:
* Max-pooling layer: slides an (f, ff,f) window over the input and stores the max value of the window in the output.
* Average-pooling layer: slides an (f, ff,f) window over the input and stores the average value of the window in the output.
they have hyperparameters such as the window size f

