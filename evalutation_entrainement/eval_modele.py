import numpy as np
import keras

classes = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping', 'drinking', 'opening microwave oven', 'Throwing trash']

test_x = np.load('C:/Users/master/Documents/Stage/dataset/oad_fusion_2img/10/40/test_x.npy')
test_y = np.load('C:/Users/master/Documents/Stage/dataset/oad_fusion_2img/10/40/test_y.npy')

tab = [0,0,0,0,0,0,0,0,0,0]

for i in range(len(test_y)):
    tab[np.argmax(test_y[i],axis=-1)] += 1
print(tab)
model = keras.models.load_model('C:/Users/master/Documents/Stage/modeles/fusion_2img/40/90_45_perso.h5')
model.summary()
print(model.evaluate(test_x,test_y))
pred_y = model.predict(test_x)
predy = []
testy = []

for i in range(pred_y.shape[0]):
    predy.append(np.argmax(np.round(pred_y[i]),axis=-1))
    testy.append(np.argmax(test_y[i],axis=-1))
predy = np.asarray(predy)
testy = np.asarray(testy)
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix(testy,predy,normalize='true'),display_labels=classes)
disp.plot()
plt.show()

