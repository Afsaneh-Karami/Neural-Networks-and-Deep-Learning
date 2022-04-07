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
7. Layer Freezing with the Functional API <br />
There are three steps:
a. Delete the top layer (the classification layer)(GOTO [alpaca_model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2))<br />
* Set include_top in base_model as False
b. Add a new classifier layer
* Train only one layer by freezing the rest of the network
* As mentioned before, a single neuron is enough to solve a binary classification problem.
c. Freeze the base model and train the newly-created classifier layer
* Set base model.trainable=False to avoid changing the weights and train only the new layer
* Set training in base_model to False to avoid keeping track of statistics in the batch norm layer
7. Create new model using the data_augmentation function defined earlier (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)line 6)<br />.
8. Compile the new model and run it for 5 epochs: (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)line 7 until 10)
9. Plot the training and validation accuracy: (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)line 11 until 34)
### Fine-tuning the Model:(GOTO [Fine-tuning the Model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2))<br />
You could try fine-tuning the model by re-running the optimizer in the last layers to improve accuracy. When you use a smaller learning rate, you take smaller steps to adapt it a little more closely to the new data. In transfer learning, the way you achieve this is by unfreezing the layers at the end of the network, and then re-training your model on the final layers with a very low learning rate. Adapting your learning rate to go over these layers in smaller steps can yield more fine details  and higher accuracy.<br />

The intuition for what's happening: when the network is in its earlier stages, it trains on low-level features, like edges. In the later layers, more complex, high-level features like wispy hair or pointy ears begin to emerge. For transfer learning, the low-level features can be kept the same, as they have common features for most images. When you add new data, you generally want the high-level features to adapt to it, which is rather like letting the network learn to detect features more related to your data, such as soft fur or big teeth.<br />

To achieve this, just unfreeze the final layers and re-run the optimizer with a smaller learning rate, while keeping all the other layers frozen.<br />

Where the final layers actually begin is a bit arbitrary, so feel free to play around with this number a bit. The important takeaway is that the later layers are the part of your network that contain the fine details (pointy ears, hairy tails) that are more specific to your problem.<br />
First, unfreeze the base model by setting base_model.trainable=True, set a layer to fine-tune from, then re-freeze all the layers before it. <br />






