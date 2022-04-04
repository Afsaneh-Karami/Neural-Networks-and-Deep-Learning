## Applying YOLO algorithm (You Only Look Once) to detect cars in the images<br />

In this file, I implemented YOLO algorithm.An algorithm that uses neural networks to provide real-time object detection. This algorithm is popular because of its speed and accuracy. Its name means "You Only Look Once" because This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.<br /> This Algorithm use two step to decrease the number of bounding boxes to desire amount.<br />
* First filter: Filtering with a Threshold on Class Scores to eliminate bounding boxes that have low probability or low score 
* Second filter: Using non-maximum suppression (NMS) for selecting the right boxes
The order of functions to make YOLO algorithm:<br />
 # Model Details<br />
* Inputs and outputs: 
The input is a batch of images, and each image has the shape (m, 608, 608, 3)<br />
The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (p_c, b_x, b_y, b_h, b_w, c). Where p_c is the probability that there is an object in the grid cell, and b_x, b_y, b_h, b_w indicate the position of bounding box in the grid cell, and c is the probability that the object is a certain class. The classes are 80. If you expand c into an 80-dimensional vector, each bounding box is then represented by 85 numbers. So, the output has tha shape (m, 19, 19, 5, 85).<br />
* Anchor Boxes:
Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the 80 different classes. So, 5 anchor boxes were chosen which is available in the file '/model_data/yolo_anchors.txt'. (GOTO [zero_pad link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Convolutional%20Neural%20Network/zero_pad)) 



