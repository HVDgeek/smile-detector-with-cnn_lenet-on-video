import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from hvdev.nn.cnn import LeNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np 
import cv2
import os
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = "path to the dataset")
ap.add_argument('-m', '--model', required = True, help = 'path to the output model')
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in sorted(list(paths.list_images(args['dataset']))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image , width = 28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

data = np.array(data , dtype = 'float')/255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotal = labels.sum(axis = 0)
classWeight = classTotal.max()/classTotal

(trainX , testX, trainY, testY) = train_test_split(data , labels , test_size = 0.20 , 
    stratify = labels , random_state = 42)

print('[INFO] compiling model')
model = LeNet().build(height = 28 , width = 28 , depth = 1 , classes = 2)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print('[INFO] training network...')
H = model.fit(trainX , trainY , validation_data = (testX , testY ), epochs = 15, batch_size = 64,
    class_weight = classWeight, verbose =1 )

print('[INFO] serializing model...')
model.save(args['model'])

print('[INFO] Evaluating model...')
predictions = model.predict(testX , batch_size = 64).argmax(axis = 1)
print(classification_report(testY.argmax(axis = 1), predictions, target_names = le.classes_))

plt.style.use('ggplot')
plt.plot(np.arange(0 , 15), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0 , 15), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0 , 15), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0 , 15), H.history['val_acc'], label = 'val_acc')
plt.title('training Accuracy/loss ')
plt.xlabel('#epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('output/plot.png')
plt.show()