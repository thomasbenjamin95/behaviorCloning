import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

##set 1 - the data file provided by udacity -  appending to images and  measurements list
lines = []

with open('data/Set 1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:

        lines.append(line)

#set 2- data from two lap-  appending to images and  measurement list
lines2= []
with open('data/Stay_center/driving_log.csv') as csvfile2:
    reader2 = csv.reader(csvfile2)
    for line2 in reader2:
        lines2.append(line2)

#recovery -- recovery dataset - appending to images and  measurement list
lines_recovery= []
with open('data/recovery/driving_log.csv') as csv_recovery:
    reader_recovery_csv= csv.reader(csv_recovery)
    for line_recovery in reader_recovery_csv:
        lines_recovery.append(line_recovery)
#stay center dataset appending to images and  measurement list
lines_set2 = []
with open('data/Set 2/driving_log.csv') as set2_csv_file:
    set2_csv = csv.reader(set2_csv_file)
    for line_set2 in set2_csv:
        lines_set2.append(line_set2)



images,measurements = [],[]
augmented_images,augmented_measurements =[],[]

for line in lines:

    for i in range(3):
        source = line[i]
        file_name = source.split('\\')[-1]
        image = cv2.imread('data/Set 1/'+ file_name)
        images.append(image)
        if i == 0:

            measurements.append(float(line[3]))
        if i == 1:
            measurements.append(float(line[3])+ 0.2)
        if i == 2:
            measurements.append(float(line[3])- 0.2)

augmented_images.extend(images)
augmented_measurements.extend(measurements)








images,measurements = [],[]
for set2_line in lines_set2:
    for i in range(3):
        source = set2_line[i]
        file_name = source.split('\\')[-1]
        image = cv2.imread('data/Set 2/IMG/' + file_name)
        images.append(image)
        if i == 0:
            measurements.append(float(set2_line[3]))
        if i == 1:
            measurements.append(float(set2_line[3]) + 0.2)
        if i == 2:
            measurements.append(float(set2_line[3]) - 0.2)
augmented_images.extend(images)
augmented_measurements.extend(measurements)

tot_augmented_images,tot_augmented_measurements= [],[]

for image in augmented_images:
    tot_augmented_images.append(image)
    tot_augmented_images.append(cv2.flip(image,1))


for q in augmented_measurements:
    tot_augmented_measurements.append(q)
    tot_augmented_measurements.append(q*-1.0)




X_train = np.array(tot_augmented_images)
y_train = np.array(tot_augmented_measurements)

print(X_train.shape)





from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout,Activation
from keras.layers import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,(5,5), activation='relu',strides = (2,2)))
model.add(Convolution2D(36,(5,5), activation='relu',strides = (2,2)))
model.add(Convolution2D(48,(5,5),activation='relu',strides=(2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,(3,3),activation='relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam' )
a = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, epochs= 5)

print(a.history['val_loss'])

plt.plot(a.history['loss'])
plt.plot(a.history['val_loss'])
plt.legend(['training set','validation set'], loc='upper right')
plt.show()






model.save('model.h5')