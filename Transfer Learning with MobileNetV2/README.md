# Using transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier:
Transfer learning means using a pre-train model for another problem that is different but related. Two problems have some common features. In this code, I used a pre-trained CNN named MobileNetV2 to classify an Alpaca/Not Alpaca images. The MobileNetV2 is fast and efficient for this classification. It's been pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes. I need to change the last layer and its activation function in order to match with binary classification (which explain in step 7).<br />
The order of functions to make transfer learning on a pre-trained CNN:<br />
1. Create the dataset and split it into training and validation sets (GOTO [Create Dataset link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Transfer%20Learning%20with%20MobileNetV2/Create%20Dataset))<br /><br />
2. Using dataset.prefetch for data preprocessing (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)lines 2, and 3)<br /> <br />
Note: 
* Using prefetch() prevents a memory bottleneck that can occur when reading from disk. It sets aside some data and keeps it ready for when it's needed, by creating a source dataset from your input data, applying a transformation to preprocess it, then iterating over the dataset one element at a time. Because the iteration is streaming, the data doesn't need to fit into memory.<br /> 
* You can set the number of elements to prefetch manually, or you can use tf.data.experimental.AUTOTUNE to choose the parameters automatically. <br /><br /> 
3. Data augmentation (GOTO [data_augmenter link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)):<br /> 
Implement a function for data augmentation. Use a Sequential keras model composed of:<br /> 
* RandomFlip('horizontal')<br /> 
* RandomRotation(0.2)<br /> <br />
4. Apply the first tool from the MobileNet application in TensorFlow, to normalize input.<br /> 
Since I am using a pre-trained model that was trained on the normalization values [-1,1], it's best practice to reuse that standard with tf.keras.applications.mobilenet_v2.preprocess_input. (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)line 4)<br /> <br />
5. Using MobileNetV2 model for Transfer Learning<br />
MobileNetV2 was trained on ImageNet and is optimized to run on mobile and other low-power applications. It's 155 layers deep and very efficient for object detection and image segmentation tasks, as well as classification tasks like this one. The architecture has three defining characteristics:<br /> 
* Depthwise separable convolutions<br />
* Thin input and output bottlenecks between layers<br />
* Shortcut connections between bottleneck layers<br />
MobileNetV2 uses depthwise separable convolutions as efficient building blocks. Traditional convolutions are often very resource-intensive, and depthwise separable convolutions can reduce the number of trainable parameters and operations and also speed up convolutions in two steps:<br />
a. The first step calculates an intermediate result by convolving each of the channels independently. This is the depthwise convolution.<br /> 
b. In the second step, another convolution merges the outputs of the previous step into one. This gets a single result from a single feature at a time and then is applied to all the filters in the output layer. This is the pointwise convolution.<br />
Each block consists of an inverted residual structure with a bottleneck at each end. These bottlenecks encode the intermediate inputs and outputs in a low-dimensional space and prevent non-linearities from destroying important information.<br />
The shortcut connections, which are similar to the ones in traditional residual networks, serve the same purpose of speeding up training and improving predictions. These connections skip over the intermediate convolutions and connect the bottleneck layers. <br /><br />
6. Train the base model by using all the layers from the pre-trained model.<br />
I also loaded the pre-trained weights from ImageNet by specifying weights='imagenet'.(GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2)line 5)<br /><br />
7. Layer Freezing with the Functional API (GOTO [alpaca_model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2))<br />
There are three steps:<br />
* Delete the top layer (the classification layer)
Set include_top in base_model as False<br />
* Add a new classifier layer<br />
Train only one layer by freezing the rest of the network<br />
As mentioned before, a single neuron is enough to solve a binary classification problem.<br />
* Freeze the base model and train the newly-created classifier layer<br />
Set base model.trainable=False to avoid changing the weights and train only the new layer<br />
Set training in base_model to False to avoid keeping track of statistics in the batch norm layer<br /><br />
8. Create a new model using the data_augmentation function defined earlier (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2) line 6).<br /><br />
9. Compile the new model and run it for 5 epochs: (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2) line 7 until 10)<br /><br />
10. Plot the training and validation accuracy: (GOTO [main program link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2) line 11 until 34)<br />
### Fine-tuning the Model:(GOTO [Fine-tuning the Model link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Transfer%20Learning%20with%20MobileNetV2))<br />
You could try fine-tuning the model by re-running the optimizer in the last layers to improve accuracy by using smaller learning rate. A smaller learning rate means take smaller steps to adapt it a little more closely to the new data. In transfer learning, the way you achieve this is by unfreezing the layers at the end of the network and then re-training your model on the final layers with a very low learning rate. Adapting your learning rate to go over these layers in smaller steps can yield more fine details and higher accuracy.<br />
The intuition for what's happening: when the network is in its earlier stages, it trains on low-level features, like edges. In the later layers, more complex, high-level features like wispy hair or pointy ears begin to emerge. For transfer learning, the low-level features can be kept the same, as they have common features for most images. When you add new data, you generally want the high-level features to adapt to it, which is rather like letting the network learn to detect features more related to your data, such as soft fur or big teeth.<br />
To achieve this, just unfreeze the final layers and re-run the optimizer with a smaller learning rate, while keeping all the other layers frozen.<br />
First, unfreeze the base model by setting base_model, set a layer to fine-tune from, then re-freeze all the layers before it. <br />






