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
Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the 80 different classes. So, 5 anchor boxes were chosen which is available in the file 'yolo_anchors.txt'. (GOTO [yolo_anchors link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Car%20detection%20with%20YOLO%20%20algorithm/Datasets)) <br />
The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85) which is shown in the below picture.<br />
<img width="824" alt="architecture" src="https://user-images.githubusercontent.com/78735911/161499356-09d1105d-0678-4b80-b05d-1983ef40e8aa.png">
Since you're using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.
For simplicity, you'll flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425).<br />
<img width="791" alt="flatten" src="https://user-images.githubusercontent.com/78735911/161508295-7041650c-f266-4e0c-adf9-1ad3578fb98a.png"><br /><br />

 # First filter: Filtering with a threshold on class scores (GOTO [yolo_filter_boxes](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Car%20detection%20with%20YOLO%20%20algorithm/yolo_filter_boxes))<br /> 
* Class score 
Now, for each box (of each cell) you'll compute the following element-wise product and extract a probability that the box contains a certain class.
The class score is  𝑠𝑐𝑜𝑟𝑒𝑐,𝑖=𝑝𝑐×𝑐𝑖 : the probability that there is an object  𝑝𝑐  times the probability that the object is a certain class  𝑐𝑖 . And then You're going to first apply a filter by thresholding, meaning you'll get rid of any box for which the class "score" is less than a chosen threshold. 
<img width="825" alt="probability_extraction" src="https://user-images.githubusercontent.com/78735911/161508983-b2b9fe38-9958-49f8-8d28-8616e6ecfc4b.png">

1. First rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:<br />
* box_confidence: tensor of shape  (19,19,5,1)  containing  𝑝𝑐  (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.<br />
* boxes: tensor of shape  (19,19,5,4)  containing the midpoint and dimensions  (𝑏𝑥,𝑏𝑦,𝑏ℎ,𝑏𝑤)  for each of the 5 boxes in each cell.<br />
* box_class_probs: tensor of shape  (19,19,5,80)  containing the "class probabilities"  (𝑐1,𝑐2,...𝑐80)  for each of the 80 classes for each of the 5 boxes per cell.<br />
2. Compute box scores by doing the following elementwise product:<br />
* box_scores = box_confidence*box_class_probs <br />
3. Use to following function to find the positio and value of the max in box_scores (set axis to -1).<br />
* box_classes = tf.math.argmax(box_scores,axis=-1)<br />
* box_class_scores = tf.math.reduce_max(box_scores,axis=-1)<br />
4. Creating a mask by using a threshold to change the numbers in box_class_scores to boolean  <br />
* filtering_mask = (box_class_scores>=threshold)<br />
5. Apply the mask to box_class_scores, boxes and box_classes<br />
* scores = tf.boolean_mask(box_class_scores,filtering_mask)<br />
* boxes = tf.boolean_mask(boxes,filtering_mask)<br />
* classes = tf.boolean_mask(box_classes,filtering_mask)<br />
 # Srecond filter:Non-max Suppression








