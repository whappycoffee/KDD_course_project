import tensorflow as tf 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train = pd.read_csv('digit-recognizer/train.csv')
test = pd.read_csv('digit-recognizer/test.csv')

### transform train and test into image/label
x_train = train.drop(['label'], axis=1).values.astype('float32') # all pixel values
y_train = train['label'].values.astype('int32') # only labels i.e targets digits
test = test.values.astype('float32')

x_train = x_train.reshape(x_train.shape[0], 28, 28) / 255.0
test = test.reshape(test.shape[0], 28, 28) / 255.0


x_train, x_test, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)
##plt.imshow(x_train[0])
##print(y_train[0])
##plt.show()#

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))#

model.compile(optimizer ="adam",
    loss ="sparse_categorical_crossentropy",
    metrics = ['accuracy'])

model.fit(x_train, y_train,epochs = 10)

val_loss, val_acc = model.evaluate(x_test, y_val)
print(val_loss,val_acc)
model.save("method1.model")
print(type(test))
model1 = tf.keras.models.load_model("method1.model")
predictions = model1.predict(test)

f = open("model1results.csv","w+")
f.write("ImageId,Label"+"\n")
i = 0
for nums in predictions:
    i+=1
    f.write(str(i)+","+str(np.argmax(predictions[i-1]))+'\n')
f.close()
#results = np.argmax(predictions)
#print(predictions[0])
#print(predictions.shape)

#print(np.argmax(predictions[0]))
#plt.imshow(test[0])
#plt.show()


