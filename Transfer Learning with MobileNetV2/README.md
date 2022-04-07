# Using transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier:
In this code, I used a pre-trained CNN named MobileNetV2 to classify an Alpaca/Not Alpaca images.A pre-trained model is a network that's already been trained on a large dataset and saved, which allows you to use it to customize your own model cheaply and efficiently. MobileNetV2 was designed to provide fast and computationally efficient performance. It's been pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes. <br />
The order of functions to make an transfer learning on a pre-trained CNN :<br />
1. Create the Dataset and Split it into Training and Validation Sets (GOTO [Create Dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Transfer%20Learning%20with%20MobileNetV2/Create%20Dataset))<br />
2.Using dataset.prefetch for data preprocessing (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)lines 2, and 3)<br /> 
Note: 
* You can set the number of elements to prefetch manually, or you can use tf.data.experimental.AUTOTUNE to choose the parameters automatically. <br /> 
* Using prefetch() prevents a memory bottleneck that can occur when reading from disk. It sets aside some data and keeps it ready for when it's needed, by creating a source dataset from your input data, applying a transformation to preprocess it, then iterating over the dataset one element at a time. Because the iteration is streaming, the data doesn't need to fit into memory.<br /> 
3. Data augmentation (GOTO [data_augmenter link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)):<br /> 
Implement a function for data augmentation. Use a Sequential keras model composed of 2 layers:<br /> 
* RandomFlip('horizontal')<br /> 
* RandomRotation(0.2)<br /> 
4. Apply the first tool from the MobileNet application in TensorFlow, to normalize input.<br /> 
Since I am using a pre-trained model that was trained on the normalization values [-1,1], it's best practice to reuse that standard with tf.keras.applications.mobilenet_v2.preprocess_input. (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)line 4)<br /> 
5. Using MobileNetV2 for Transfer Learning
MobileNetV2 was trained on ImageNet and is optimized to run on mobile and other low-power applications. It's 155 layers deep (just in case you felt the urge to plot the model yourself, prepare for a long journey!) and very efficient for object detection and image segmentation tasks, as well as classification tasks like this one. The architecture has three defining characteristics:<br /> 
* Depthwise separable convolutions<br />
* Thin input and output bottlenecks between layers<br />
* Shortcut connections between bottleneck layers<br />
MobileNetV2 uses depthwise separable convolutions as efficient building blocks. Traditional convolutions are often very resource-intensive, and depthwise separable convolutions are able to reduce the number of trainable parameters and operations and also speed up convolutions in two steps:<br />
a. The first step calculates an intermediate result by convolving on each of the channels independently. This is the depthwise convolution.
b. In the second step, another convolution merges the outputs of the previous step into one. This gets a single result from a single feature at a time, and then is applied to all the filters in the output layer. This is the pointwise convolution, or: Shape of the depthwise convolution X Number of filters.<br />
![mobilenetv2](https://user-images.githubusercontent.com/78735911/162256874-1810d349-5edd-4154-890a-e310b6de3ef9.png)<br />
Each block consists of an inverted residual structure with a bottleneck at each end. These bottlenecks encode the intermediate inputs and outputs in a low dimensional space, and prevent non-linearities from destroying important information.<br />
The shortcut connections, which are similar to the ones in traditional residual networks, serve the same purpose of speeding up training and improving predictions. These connections skip over the intermediate convolutions and connect the bottleneck layers. <br />
6. Train your base model using all the layers from the pretrained model.<br />
Similarly to how you reused the pretrained normalization values MobileNetV2 was trained on, you'll also load the pretrained weights from ImageNet by specifying weights='imagenet'.(GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)line 5)<br />






