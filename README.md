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
driving performance of the vehicle. The datasets are: </p>
<p>● Two laps of center lane line dataset</p>
<p>● Dataset from the project resources. </p>
