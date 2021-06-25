# behaviorCloning
# <h2><u>solution Design Approach</u></h2>
<p>The overall strategy for deriving a model architecture was to drive the car
autonomously through the track.
My first step was to use a convolutional neural network model similar to the
network we used to train the images in order to identity the images with appropriate labels. I
thought this model is appropriate because the project was very similar to traffic sign classifier
project and in place of labels we used appropriate measurements.
To combat the overfitting, i modified the model so that it could reduce overfitting. In
this model i used dropout to prevent the zero valued node and relu for non linearity.
Then i used 0.2 and 0.5 values for the dropout. The input image shape to the
neural network is (160,320,3). After accepting the image with this resolution i used the
cropping2D to crop the image of 70 percent from the top and 20 percent from the bottom.
The final step was to run the simulator to see how well the car was driving around
track one. There were a few spots where the vehicle fell off the tracks and i used the other
datasets to improve behavior in these cases. I also used other two dataset to improve the
driving performance of the vehicle. The datasets are:
<br>● Two laps of center lane line dataset.
<br>● Dataset from the project resources. </p>
<h2><u>final Model Architecture</u></h2>
<p>The final model architecture(model.py line 90-112) consisted of a convolutional
neural network with the following layers and layer sizes of (160,320,3) as input. Used lamba for
normalization of the images. After normalizing the images, the images will be cropped using
cropping2D of 70 percent from the top and 20 percent from the bottom. Then i used five
convolutional2D with strides of (2,2) and filter size of 24,36,48,64 and 64. For non linearity and
to reduce overfitting both relu and dropout have been used. Then the model is flatten and fully
connected until it reaches the with value 1. Adam optimizer is used as the algorithm since we do
not want to tune the learning rates. The training model is split into validation mode l of sample
by 20 percent.The number of epochs is 5.</P>
<h2>creation of the Training Set and Training Process</h2>

<img src="![image](https://user-images.githubusercontent.com/86484259/123454015-16e40480-d5ae-11eb-9bc6-48d6f3e709ea.png)"

