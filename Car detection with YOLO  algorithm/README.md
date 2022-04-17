# Applying YOLO algorithm (You Only Look Once) to detect cars in the images<br />

In this file, I implemented the YOLO algorithm. An algorithm that uses neural networks to detect the object in the input image. The reputation of This algorithm is because of its speed and accuracy. Its name means "You Only Look Once", This algorithm "only looks once" at the image, only one forward propagation pass through the network is enough to make predictions. YoLo Algorithm applies some filters such as non-max suppression to output recognized objects together with the bounding boxes. After fitting the neural network model to the input image, there are lots of bounding boxes for each object with different probabilities, so some filters like non-max suppression are used to find the best bounding box that encompasses the object. <br /> 
This Algorithm uses two steps to decrease the number of bounding boxes to the desired amount.<br />
* First filter: Filtering with a threshold on class scores to eliminate bounding boxes with low probability or low score 
* Second filter: Using non-maximum suppression (NMS) for choosing the best boxes
The order of functions to make the YOLO algorithm:<br />
 ## Model Details<br />
* Inputs and outputs: 
The input is a batch of images with the shape (m, 608, 608, 3)<br />
The output is a list of bounding boxes for objects and the classification of objects. Each bounding box is represented by six numbers (p_c, b_x, b_y, b_h, b_w, c). Where p_c is the probability that there is an object in the grid cell, b_x, b_y, b_h, b_w indicate the position of the bounding box in the grid cell. c is the probability that the object belongs to a certain class of the  80 classes. <br />
* Anchor Boxes:
Anchor boxes are chosen by reasonable height/width ratios that can represent the 80 different classes. So, five anchor boxes were chosen which is available in the file 'yolo_anchors.txt'. (GOTO [yolo_anchors link](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/tree/main/Car%20detection%20with%20YOLO%20%20algorithm/Datasets)) <br />
The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).<br />
Since you're using 5 anchor boxes, each of the 19 x19 cells contains information about 5 boxes. Anchor boxes are defined only by their width and height.
For simplicity, you can flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425).<br />

 ## First filter: Filtering with a threshold on class scores (GOTO [yolo_filter_boxes](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Car%20detection%20with%20YOLO%20%20algorithm/yolo_filter_boxes))<br /> 
* Class score 
For each box (of each cell) you'll compute the probability that the box contains a certain class, multiplying the probability of the existence of an object in the bounding box (𝑝𝑐) by the probability of each class (𝑐𝑖) 𝑖=𝑝𝑐×𝑐𝑖. Then You're going to apply the first filter by a threshold, meaning you'll get rid of any box for which the class "score" is less than a chosen threshold. In the following you can see steps for applying first filter. <br />
1. First rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:<br />
* box_confidence: tensor of shape  (19,19,5,1)  containing  𝑝𝑐  (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.<br />
* boxes: tensor of shape  (19,19,5,4)  containing the midpoint and dimensions  (𝑏𝑥,𝑏𝑦,𝑏ℎ,𝑏𝑤)  for each of the 5 boxes in each cell.<br />
* box_class_probs: tensor of shape  (19,19,5,80)  containing the "class probabilities"  (𝑐1,𝑐2,...𝑐80)  for each of the 80 classes for each of the 5 boxes per cell.<br />
2. Compute box scores by doing the following elementwise product:<br />
* box_scores = box_confidence * box_class_probs <br />
3. Use to following function to find the position and value of the maximum in box_scores (set axis to -1).<br />
* box_classes = tf.math.argmax(box_scores,axis=-1)<br />
* box_class_scores = tf.math.reduce_max(box_scores,axis=-1)<br />
4. Creating a mask by using a threshold to change the numbers in box_class_scores to boolean.  <br />
* filtering_mask = (box_class_scores>=threshold)<br />
5. Apply the mask to box_class_scores, boxes, and box_classes<br />
* scores = tf.boolean_mask(box_class_scores,filtering_mask)<br />
* boxes = tf.boolean_mask(boxes,filtering_mask)<br />
* classes = tf.boolean_mask(box_classes,filtering_mask)<br />
 ## Second filter:Non-max Suppression (GOTO [yolo_non_max_suppression](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Car%20detection%20with%20YOLO%20%20algorithm/yolo_non_max_suppression))<br /> 
Even after first filtering, you still end up with a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS).<br />
Non-max suppression uses the very important function called "Intersection over Union", or IoU. This function makes a number by dividing the intersection of two boxes by the union of them. <br />
Some Notes about tha calculation of IoU:<br />
* This code supposed that (0,0) is the top-left corner of an bounding box, (1,0) is the upper-right corner, and (1,1) is the lower-right corner. In other words, the (0,0) origin starts at the top left corner of the image. As x increases, you move to the right. As y increases, you move down.
* A box is defined in two manner:
1. using its two corners: upper left  (𝑥1,𝑦1)  and lower right  (𝑥2,𝑦2) .To calculate the area of a rectangle, multiply its height  (𝑦2−𝑦1)  by its width  (𝑥2−𝑥1). Since  (𝑥1,𝑦1)  is the top left and  (𝑥2,𝑦2)  are the bottom right, these differences should be non-negative. <br />
2. using width and height of the box and its center position like (x,y,w,h). This format can change to the first one (GOTO [yolo_boxes_to_corners](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/new/main/Car%20detection%20with%20YOLO%20%20algorithm))<br /> 
*  Calculation of intersection area of two boxes based on the first box definition manner:<br />
Finding the coordinate of intersection area:<br />
xi1 = maximum of the x1 coordinates of the two boxes<br />
yi1 = maximum of the y1 coordinates of the two boxes<br />
xi2 = minimum of the x2 coordinates of the two boxes<br />
yi2 = minimum of the y2 coordinates of the two boxes<br />
width=max((xi2-xi1),0)<br />
height=max((yi2-yi1),0)<br />
inter_area = width * height <br />
If two boxes have no intersection then the height and width become negative so the max function shows zero as the maximum value. So, if inter_area how zero value means no intersection. <br />
* Implement non-max suppression. The key steps are:<br />
1. Select the box that has the highest score.<br />
2. Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= iou_threshold).<br />
3. Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.<br />
This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.<br />
Tensorflow used build-in function non_max_suppression to calculate all the above mentioned stages itself. This function get the list of indices corresponding to boxes that keep.<br />
* tf.image.non_max_suppression(boxes,scores,max_output_size,iou_threshold=0.5, name=None)
Gahther function show the value of the indices in refrence matrix: 
* keras.gather( reference,indices)
# Applying YOLO alghorithm on the trainig set(GOTO [yolo_eval](https://github.com/Afsaneh-Karami/Neural-Networks-and-Deep-Learning/blob/main/Car%20detection%20with%20YOLO%20%20algorithm/yolo_eval)):<br />
Implement yolo_eval() which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. YOLO's network was trained to run on 608x608 images.
 
  

  
  
  
 












