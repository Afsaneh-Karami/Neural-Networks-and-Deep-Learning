# Applying Siamese network for face verification and face recognition: <br />
In face verification, you're given two images and you have to determine if they are of the same person or not. In face recognition, you compare an input picture with a database of people's images and determine if the input image belongs to one person in the database or not.<br />
For this project, I used a pre-trained model, the Inception model from Szegedy, and implemented the weights that someone else has already trained. The FaceNet model takes a lot of data and a long time to train, so it is efficient to use a pre-train model. In the following link, the architecture of the Inception model from Szegedy et al is available. (GOTO [Inception_blocks_v2](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/Inception_blocks_v2)). By applying this model on input images I get a encoding Face Images into a 128-Dimensional Vector. <br />
 Notes about Inception_blocks_v2 neural network: <br />
* This network uses 160x160 dimensional RGB images as input. Specifically, a face image or batch of 𝑚 face images as a tensor of shape (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶)=(𝑚,160,160,3)<br />
* The output is a matrix of shape (𝑚,128) that encodes each input face image into a 128-dimensional vector.<br />
Loading the model and its pre-trained weights.(GOTO [Loading the model](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/Loading%20the%20model)) <br />
## Encoding Face Images into a 128-Dimensional Vector
To generate the encoding, the function img_to_encoding is used,  img_to_encoding(image_path, model), which runs the forward propagation of the model,Inception_blocks_v2 neural network, on the specified image.(GOTO [img_to_encoding](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/img_to_encoding)) <br /> 
The input images of Inception_blocks_v2 neural network are in shape 160x160 and my images are 96x96, thus, I need to scale them to 160x160. I add the below line in img_to_encoding to load the image in the correct size.
* img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160)) <br /><br />
A good encoding is a good one if:<br />
The encodings of two images of the same person are quite similar to each other.<br />
The encodings of two images of different persons are very different.<br />

## Face Verification
In this problem, two picture image_path and identity are compared to evaluate whether two pictures belong to the same person or not. It is done by the following steps:<br />
1.Run the database (GOTO [Run database](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/Run%20database)) to build the database (represented as a Python dictionary). This database maps each person's name to a 128-dimensional encoding of their face.<br />
2.Implement the verify() function to get the encoding (GOTO [verify](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/verify)). It checks if the picture taken by front-door camera (image_path) is the person called "identity" by the following steps:<br />
2.1 Compute the encoding of the image from image_path.<br />
2.2 Compute the distance between this encoding and the encoding of the identity image stored in the database.<br />
2.3 Open the door if the distance is less than 0.7, otherwise do not open it.<br />
Note: I used L2 distance np.linalg.norm to evaluate the distance between two images encoding <br />
Note: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold of 0.7.
## Face Recognition
In this problem an input image is compared with other pictures in the database and if the input image belongs to a person in the database the door is open. It is done by the following steps:
1. Run the database (GOTO [Run database](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/Run%20database)) to build the database (represented as a Python dictionary). This database maps each person's name to a 128-dimensional encoding of their face.<br />
2. Implement the who_is_it() function (GOTO [who_is_it](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/who_is_it)). This function include the following steps: <br />
2.1 Compute encoding of the image from image_path<br />
2.2 Find the encoding from the database that has the smallest distance from the target encoding.<br />
2.3 Initialize the min_dist variable to a large enough number (100). This helps you keep track of the closest encoding to the input's encoding.<br />
2.4 Loop over the database dictionary's names and encodings. To loop use for (name, db_enc) in database.items().<br />
2.5 Compute the L2 distance between the target "encoding" and the current "encoding" from the database. If this distance is less than the min_dist, then set min_dist to dist, and identity to name.<br />
## The Triplet Loss
Note: Since I am using a pre-trained model, I won't need to implement the triplet loss function to determine parameters that minimize the cost function. But because of its importance, I bring it into this code.
Training will use triplets of images  (A, P, N) when:<br />
A is an "Anchor" image--a picture of a person.<br />
P is a "Positive" image--a picture of the same person as the Anchor image.<br />
N is a "Negative" image--a picture of a different person than the Anchor image.<br />
These triplets are picked from the training dataset. For i -th training example triplet is (A(i),P(i),N(i)).
I need to make sure that an image 𝐴(𝑖) of an individual is closer to the positive  𝑃(𝑖)  than to the negative image 𝑁(𝑖) by at least a margin 𝛼. So, I need to minimize the following "triplet cost":<br />
![Untitled](https://user-images.githubusercontent.com/78735911/162184994-bb32758d-914a-497e-aae3-1fb4c0a29f8b.png)<br />
Here, the notation " [𝑧]+ " is used to denote  𝑚𝑎𝑥(𝑧,0) .
* Notes:<br />
The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet, which should be small.<br />
The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, which should be relatively large. <br />
𝛼  is called the margin. It's a hyperparameter that you pick manually. I choosed 𝛼=0.2 .<br />
* Implement the triplet loss (GOTO [triplet_loss](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Face%20verification%20and%20face%20recognition/triplet_loss)) by 4 steps:<br />
1. Compute the distance between the encodings of "anchor" and "positive" <br />
2. Compute the distance between the encodings of "anchor" and "negative" <br />
3. Compute the formula J per training example <br />
4. Compute the full formula J by taking the max with zero and summing over the training examples <br />






