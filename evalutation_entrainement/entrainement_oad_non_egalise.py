import numpy as np
from keras.applications.vgg16 import VGG16
import keras
from keras.layers import Dense,Reshape,Input,BatchNormalization,Dropout,Concatenate,Flatten,Lambda,Resizing,CuDNNLSTM,concatenate,Layer
from keras import  Model,layers
from keras.callbacks import ModelCheckpoint, EarlyStopping


input_shape= (25,40, 3)

train_x = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug\\train_x.npy',allow_pickle=True)
train_y = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug\\train_y.npy',allow_pickle=True)
test_x = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug\\test_x.npy',allow_pickle=True)
test_y = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug\\test_y.npy',allow_pickle=True)

train_x = keras.applications.vgg16.preprocess_input(train_x)
test_x = keras.applications.vgg16.preprocess_input(train_x)

inp=Input(np.asarray(input_shape))
# output = inp
output = Resizing(224, 224)(inp)
model = VGG16()
for layer in model.layers:
    layer.trainable = False

output = model(output)

output = Dense(4096)(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(4096)(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(11, activation='softmax',kernel_regularizer='l1')(output)
model = Model(inp, output)
model.summary()


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

filepath ="oad.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks = [checkpoint,EarlyStopping(monitor='val_loss', patience=100,)]
# model = keras.models.load_model('best.hdf5')
history=model.fit(test_x, test_y,
              epochs=100,
              batch_size=32,
              verbose=1,
              validation_split=0.25,
              callbacks=callbacks)
model.load_weights(filepath)
keras.models.save_model(model, 'VGG16_OAD.h5')
# new_model = keras.models.load_model(base_path+'VGG16_UOW_l1.h5')
print(model.evaluate(test_x,test_y))