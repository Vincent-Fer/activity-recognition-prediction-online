import numpy as np
from keras.applications.vgg16 import VGG16
import keras
from keras.layers import Dense,Input,BatchNormalization,Dropout,Resizing,Flatten
from keras import  Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


input_shape= (25,40,3)

train_x = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug_x\\1200\\train_x.npy')
train_y = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug_x\\1200\\train_y.npy')
test_x = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug_x\\1200\\test_x.npy')
test_y = np.load('C:\\Users\\master\\Documents\\Stage\\dataset\\oad_aug_x\\1200\\test_y.npy')

for i in range(211,216):
    rand_state = np.random.RandomState(i)
    rand_state.shuffle(train_x)
    rand_state.seed(i)
    rand_state.shuffle(train_y)

for i in range(111,116):
    rand_state = np.random.RandomState(i)
    rand_state.shuffle(test_x)
    rand_state.seed(i)
    rand_state.shuffle(test_y)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

inp=Input(np.asarray(input_shape))
# output = inp
output = Resizing(32, 32)(inp)
model = VGG16(include_top=False,input_shape=(32,32,3))
for layer in model.layers:
    layer.trainable = True
output = model(output)
output = Flatten()(output)
output = Dense(512)(output)
output = BatchNormalization()(output)
# output = Dropout(0.5)(output)
# output = Dense(512)(output)
# output = BatchNormalization()(output)
# output = Dropout(0.5)(output)
output = Dense(11, activation='softmax')(output)
model = Model(inp, output)
model.summary()


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# filepath ="oad.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
# callbacks = [checkpoint,EarlyStopping(monitor='val_loss', patience=100,)]
# model = keras.models.load_model('best.hdf5')
history=model.fit(train_x, train_y,
              epochs=100,
              batch_size=64,
              verbose=1,
              validation_split=0.25)
# model.load_weights(filepath)
keras.models.save_model(model, 'VGG16_OAD.h5')
# new_model = keras.models.load_model(base_path+'VGG16_UOW_l1.h5')
print(model.evaluate(test_x,test_y))