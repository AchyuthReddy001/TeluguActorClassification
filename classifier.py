import numpy as np
import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#Building sequential or pipeline

classifier=Sequential()

#convolutional layes with relu

#layer1
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

#layer2
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#layer3
classifier.add(Conv2D(64,(3,3),activation='relu'))

#layer4
classifier.add(Conv2D(128,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#layer5
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#layer6
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattern layer
classifier.add(Flatten())

#Dense layers/full connected layers/hiddenlayes
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
#outputlayer
classifier.add(Dense(units=5,activation='softmax'))

#compile of cnn
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

#fitting data to cnn
traing_datagen=ImageDataGenerator(rescale=1./255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


train_set=traing_datagen.flow_from_directory("E:/AI-Projects/Classification_Using_CNN/dataset",
                                             target_size=(64,64),
                                             batch_size=32,
                                             class_mode='categorical')
test_set=test_datagen.flow_from_directory("E:/AI-Projects/Classification_Using_CNN/dataset",
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='categorical')
print(train_set.class_indices)

#fit the model
classifier.fit(train_set,
               epochs=100,
               validation_data=test_set
               )

classifier.save('classifermodel.h5')
print("model saved to disk")


'''
#predict
cata=['AlluArjun','MaheshBabu','NTR','PavanKalyan','Prabhas']
test_img=image.load_img("test5.jpg",target_size=(64,64))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
res=classifier.predict(test_img)
final=classifier.predict_classes(test_img)
print(res)
if final[0]==0:
    print("AlluArjun")
elif final[0]==1:
    print("MaheshBabu")
elif final[0]==2:
    print("NTR")
elif final[0]==3:
    print("PavanKalyan")
else:
    print("Prabhas")
'''