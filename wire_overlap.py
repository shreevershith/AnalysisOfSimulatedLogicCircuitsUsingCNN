# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize the cnn
classifier = Sequential()

#Step-1 convolution
classifier.add(Convolution2D(32 ,3 ,3 , input_shape = (64 ,64 ,3) , activation = 'relu'))

#step 2 pooling
classifier.add(MaxPooling2D(pool_size =(2 ,2)))


#second convolution network
classifier.add(Convolution2D(32 ,3 ,3  , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2 ,2)))


#step3 flattening
classifier.add(Flatten())

#step4
classifier.add(Dense(output_dim =128 , activation = 'relu' ))
classifier.add(Dense(output_dim =1 , activation = 'sigmoid' ))

#compile
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#part-2 
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Wireoverlap/train_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
            'Wireoverlap/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=50,
        epochs=5,
        validation_data=test_set,
        validation_steps=20)

#making the prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Wireoverlap/prediction/AfterWireOverlap1.jpg' , target_size =(64 ,64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction =  "wire doesn't overlap"
else:
    prediction =  'wire is overlapped!'

print(prediction)


