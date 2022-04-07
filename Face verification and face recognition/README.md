# Applying Siamese network for face verification and triplet loss for face recognition: <br />
In Face Verification, you're given two images and you have to determine if they are of the same person or not. In face recognition you compare an input picture with a database of people images and determine if the input image is belong to one person in database or not.<br />
The FaceNet model takes a lot of data and a long time to train. So I used a pretrained model, I'll load weights that someone else has already trained. The network architecture follows the Inception model from Szegedy et al. By applying this model on your input image you can get a encoding Face Images into a 128-Dimensional Vector(GOTO [Using a ConvNet to Compute Encodings](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/Inception_blocks_v2)) <br />
 Notes about Inception_blocks_v2 neural network: <br />
* This network uses 160x160 dimensional RGB images as its input. Specifically, a face image (or batch of 𝑚 face images) as a tensor of shape (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶)=(𝑚,160,160,3)
* The output is a matrix of shape (𝑚,128) that encodes each input face image into a 128-dimensional vector.
## Encoding Face Images into a 128-Dimensional Vector
As mentioned before I used Inception_blocks_v2 model to get encoding of 128-dimensional vector. The input images are originally of shape 96x96, thus, I need to scale them to 160x160. Loading the model and its pretrained weights.(GOTO [Loading the model](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/Loading%20the%20model)) <br />
To generate the encoding, you'll use img_to_encoding(image_path, model), which runs the forward propagation of the model on the specified image.This is done in the img_to_encoding() function.(GOTO [img_to_encoding](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/img_to_encoding)) <br />
So, an encoding is a good one if:<br />
The encodings of two images of the same person are quite similar to each other.<br />
The encodings of two images of different persons are very different.<br />


