import os
import zipfile
import random
import tensorflow as tf
import shutil
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
# local_zip = 'tmp4/kagglecatsanddogs_3367a.zip'
# zip_ref = zipfile.ZipFile(local_zip,'r')
# zip_ref.extractall('tmp4')
# zip_ref.close()
print(len(os.listdir('tmp4/PetImages/Cat')))
print(len(os.listdir('tmp4/PetImages/Dog')))
try:
    os.mkdir('tmp4/cats-v-dogs')
    os.mkdir('tmp4/cats-v-dogs/training')
    os.mkdir('tmp4/cats-v-dogs/testing')
    os.mkdir('tmp4/cats-v-dogs/training/cats')
    os.mkdir('tmp4/cats-v-dogs/training/dogs')
    os.mkdir('tmp4/cats-v-dogs/testing/cats')
    os.mkdir('tmp4/cats-v-dogs/testing/dogs')
except OSError:
    pass
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    source_files_list = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
    for file_number in range(len(source_files_list)):
        file_source = os.path.join(SOURCE,source_files_list[file_number])
        file_training = os.path.join(TRAINING,source_files_list[file_number])
        file_tasting = os.path.join(TESTING,source_files_list[file_number])
        size = os.path.getsize(file_source)
        if (file_number)<(len(source_files_list)*SPLIT_SIZE):
            if size>0:
                copyfile(file_source, file_training)
        elif size>0:
            copyfile(file_source, file_tasting)


CAT_SOURCE_DIR = "tmp4/PetImages/Cat/"
TRAINING_CATS_DIR = "tmp4/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "tmp4/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "tmp4/PetImages/Dog/"
TRAINING_DOGS_DIR = "tmp4/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "tmp4/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
print(len(os.listdir('tmp4/cats-v-dogs/training/cats/')))
print(len(os.listdir('tmp4/cats-v-dogs/training/dogs/')))
print(len(os.listdir('tmp4/cats-v-dogs/testing/cats/')))
print(len(os.listdir('tmp4/cats-v-dogs/testing/dogs/')))
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
TRAINING_DIR = os.path.join('tmp4/cats-v-dogs/training/')
train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,batch_size=125,class_mode='binary',target_size=(300,300))

VALIDATION_DIR = os.path.join('tmp4/cats-v-dogs/testing/')
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE
# VALIDATION GENERATOR.
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,batch_size=125,class_mode='binary',target_size=(300,300))
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=180,
                              epochs=15,
                              validation_steps=20,
                              verbose=1)
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')
plt.legend()
plt.show()

