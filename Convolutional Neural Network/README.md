Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
as illustrated in below.
<img width="574" alt="PAD" src="https://user-images.githubusercontent.com/78735911/144033668-5a135959-337a-4edd-b3cf-6f0d7e25e299.png">
The main benefits of padding are:
1. It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers.
2. It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.

    
