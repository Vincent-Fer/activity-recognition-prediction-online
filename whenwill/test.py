import tensorflow as tf
import os
import time
import numpy as np
import PySimpleGUI as sg

classes = ['sweeping','gargling', 'opening_cupboard', 'washing_hands', 'eating', 'writing', 'wiping', 'drinking', 'opening_microwave_oven', 'throwing_trash']
### OAD 10 fr
classes_fr = ['balayer', 'se gargariser', 'ouvrir le placard', 'se laver les mains', 'manger', 'écrire', 'essuyer','boire','ouvrir le four à micro-ondes', 'Jeter des ordures']

path_to_activite = "../dataset/oad_whenwill/npy/"

### ACTIVITES
activites = [0,1,10,31,32,33,38,42,46,58]

def calc_moyenne():
    path = "./data/groundTruth/"
    fichiers = os.listdir(path)
    moy = 0
    moy_fin = 0
    cpt = 0
    num_tot = 50
    obs = []
    pre = []
    act= []
    activ_obs = []
    for fichier in fichiers:
        sous_obs = []
        sous_pre = []
        f = open(path+fichier,'r')
        contenu = f.readlines()
        activ = [0,0,0,0,0,0,0,0,0,0]
        l_activ_obs = []
        l_activ_pre = []
        if int(fichier.split(".")[0]) in activites:
            for i in range(0,num_tot):
                tab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                tab[classes.index(contenu[i].replace("\n",""))] = 1
                l_activ_obs.append(classes_fr[classes.index(contenu[i].replace("\n", ""))])
                sous_obs.append(np.asarray(tab))
                moy += 1
            for i in range(num_tot,num_tot + 100):
                tab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                tab[classes.index(contenu[i].replace("\n", ""))] = 1
                l_activ_pre.append(classes_fr[classes.index(contenu[i].replace("\n", ""))])
                sous_pre.append(np.asarray(tab))
                moy_fin += 1
            sous_obs = np.asarray(sous_obs)
            sous_pre = np.asarray(sous_pre)
            np.save(path_to_activite + "obs/" + fichier.split(".")[0], l_activ_obs)
            activ_obs.append(l_activ_obs)
            np.save(path_to_activite + "pre/" + fichier.split(".")[0], l_activ_pre)
            activ[activites.index(int(fichier.split(".")[0]))] = 1
            act.append(activ)
            obs.append(np.asarray(sous_obs))
            pre.append(sous_pre)
            cpt+=1
        f.close()
    act = np.asarray(act)
    obs = np.asarray(obs)
    pre = np.asarray(pre)
    moy = moy / cpt
    moy_fin = moy_fin / cpt
    return int(moy), int(moy_fin), obs, pre, act, np.asarray(activ_obs)

def build_model_pred(nb_lig,nb_fin):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16,(2,2),padding='same', activation='relu', input_shape=(int(nb_lig), 10, 1)))
    model.add(tf.keras.layers.MaxPool2D((1,1)))
    model.add(tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(int(nb_lig), 10, 1)))
    model.add(tf.keras.layers.MaxPool2D((1, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(int(nb_fin)*10, activation='relu'))
    model.add(tf.keras.layers.Reshape((int(nb_fin),10,1)))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(),
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def build_model_rec_activite(nb_lig):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (2, 2), padding='same', activation='relu', input_shape=(int(nb_lig), 10, 1)))
    model.add(tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(len(activites)*4,activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(len(activites), kernel_regularizer="l1"))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.003)
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(),
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
nb_lig, nb_fin, obs, pre, act, act_obs = calc_moyenne()
print(obs[0])
s_tab = [[0,0,0,0,0,0,0,0,0,0]]
new_obs = np.concatenate((obs[0][1:50],s_tab))
print(new_obs)
# liste_futur = [[sg.Table(values=[],headings=['prédiction'], key='-list_action_pred-', num_rows=50)]]
# layout = [
#     [sg.Column(liste_futur)]
# ]
# window = sg.Window("Image", layout).finalize()
# liste_action_1 = np.load(path_to_activite + 'obs/' + str(42) + ".npy")
# l_act_1 = []
# for i in range(len(liste_action_1)):
#     l_act_1.append([liste_action_1[i]])
# window['-list_action_pred-'].update(values=l_act_1)
# window.refresh()
# while True:
#     i=1
# model_pred = build_model_pred(nb_lig,nb_fin)
# model_activite = build_model_rec_activite(nb_lig)
# # #
# obs = obs.reshape(len(activites),50,10,1)
# pre = pre.reshape(len(activites),nb_fin,10,1)
# # print(obs.shape)
# # print(pre.shape)
# # model_pred.fit(obs,pre,batch_size=1,epochs=5000)
# # model_pred.save('./pred_.h5')
# # model_activite.fit(obs,act,batch_size=1,epochs=1000)
# # model_activite.save('./activite_.h5')
# #model_pred = tf.keras.models.load_model('pred_.h5')
# model_activite = tf.keras.models.load_model('activite_100_10.h5')
# y_pred = np.asarray(model_pred.predict(obs))
# activite_pred = np.asarray(model_activite.predict(obs))
# new_pred = []
# new_act = []
# for i in range(y_pred.shape[0]):
#     s_pred = []
#     for j in range(y_pred[i].shape[0]):
#         ss_pred = [0,0,0,0,0,0,0,0,0,0]
#         ss_pred[np.argmax(y_pred[i][j],axis=0)[0]] = 1
#         s_pred.append(ss_pred)
#     new_pred.append(s_pred)
# y_pred = np.asarray(new_pred)
# moy = 0
# for i in range(y_pred.shape[0]):
#     cpt_a = 0
#     cpt_b = 0
#     for j in range(y_pred[i].shape[0]):
#         if np.argmax(pre[i][j],axis=0)[0] == np.argmax(y_pred[i][j],axis=0):
#             cpt_a+=1
#         cpt_b+=1
#     pourc = cpt_a / cpt_b * 100
#     print("Partie " + str(i))
#     print(pourc)
#     moy += pourc
# print("Moyenne totale")
# print(moy / len(activites))
# acc = 0
# for i in range(activite_pred.shape[0]):
#     print(np.argmax(activite_pred[i],axis=-1))
#     print(np.argmax(act[i],axis=-1))
#     if np.argmax(activite_pred[i],axis=-1) == np.argmax(act[i],axis=-1):
#         acc+=1
# print("Activite precision")
# print(acc/len(activites)*100)


