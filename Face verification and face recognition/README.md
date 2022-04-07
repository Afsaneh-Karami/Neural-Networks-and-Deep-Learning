# Applying Siamese network for face verification and triplet loss for face recognition: <br />
In Face Verification, you're given two images and you have to determine if they are of the same person.By using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person.The FaceNet model takes a lot of data and a long time to train. So following the common practice in applied deep learning, you'll load weights that someone else has already trained. The network architecture follows the Inception model from Szegedy et al.



In this file, I implemented the YOLO algorithm. An algorithm that uses neural networks to provide real-time object detection. This algorithm is popular because of its speed and accuracy. Its name means "You Only Look Once", This algorithm "only looks once" at the image.  It requires only one forward propagation pass through the network to make predictions. YoLo Algorithm applies some filters such as non-max suppression to output recognized objects together with the bounding boxes.<br /> 
This Algorithm uses two steps to decrease the number of bounding boxes to the desired amount.<br />
* First filter: Filtering with a Threshold on Class Scores to eliminate bounding boxes that have low probability or low score 
* Second filter: Using non-maximum suppression (NMS) for selecting the best boxes
The order of functions to make the YOLO algorithm:<br />
 ## Model Details<br />
* Inputs and outputs: 
The input is a batch of images, and each image has the shape (m, 608, 608, 3)<br />
The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (p_c, b_x, b_y, b_h, b_w, c). Where p_c is the probability that there is an object in the grid cell, b_x, b_y, b_h, b_w indicate the position of the bounding box in the grid cell.  c is the probability that the object belongs to a certain class of the  80 classes. <br />
* Anchor Boxes:
Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the 80 different classes. So, five anchor boxes were chosen which is available in the file 'yolo_anchors.txt'. (GOTO [yolo_anchors link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Car%20detection%20with%20YOLO%20%20algorithm/Datasets)) <br />
