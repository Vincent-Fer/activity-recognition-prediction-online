import numpy as np
import os

train_sub = [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51, 54, 57, 58]
test_sub  = [0, 10, 13, 17, 21, 26, 27, 28, 29, 36, 40, 41, 42, 43, 44, 45, 52, 53, 55, 56]

actions = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping', 'drinking','opening microwave oven', 'Throwing trash']

train_squel = []
test_squel = []
train_num_fichier = []
test_num_fichier = []
train_num_activite = []
test_num_activite = []
train_img_encode = []
test_img_encode = []
train_action = []
test_action = []

window = 40

chemin = "../dataset/OAD/data/"

liste = os.listdir(chemin)

# for num_train in train_sub:
#     os.makedirs("./img/40/train/" + str(num_train))
# for num_test in test_sub:
#     os.makedirs("./img/40/test/" + str(num_test))

# for dossier in liste:
#     sous_chemin = chemin + dossier + "/skeleton/"
#     sous_liste = os.listdir(sous_chemin)
#     squel_dos = []
#     num_squel = []
#     for fichier in sous_liste:
#         f = open(sous_chemin+fichier,"r")
#         liste_lignes = f.readlines()
#         f.close()
#         nb_lignes = len(liste_lignes)
#         if nb_lignes == 25:
#             num_squel.append(int(fichier.split(".")[0]))
#             squelette = []
#             for i in range(nb_lignes):
#                 sep_ligne = liste_lignes[i].split("\n")[0].split(" ")
#                 x = float(sep_ligne[0])
#                 y = float(sep_ligne[1])
#                 z = float(sep_ligne[2])
#                 coord = [x,y,z]
#                 squelette.append(coord)
#             squel_dos.append(squelette)
#     dossier = int(dossier)
#     if dossier in train_sub:
#         train_squel.append(squel_dos)
#         train_num_fichier.append(num_squel)
#         train_num_activite.append(dossier)
#     elif dossier in test_sub:
#         test_squel.append(squel_dos)
#         test_num_fichier.append(num_squel)
#         test_num_activite.append(dossier)
#     else:
#         print(dossier)
# train_squel = np.asarray(train_squel,dtype=object)
# test_squel = np.asarray(test_squel,dtype=object)
# train_num_fichier = np.asarray(train_num_fichier,dtype=object)
# test_num_fichier = np.asarray(test_num_fichier,dtype=object)
# train_num_activite = np.asarray(train_num_activite,dtype=object)
# test_num_activite = np.asarray(test_num_activite,dtype=object)
#
# np.save("./data/train_squel.npy",train_squel)
# np.save("./data/test_squel.npy",test_squel)
# np.save("./data/train_num_fichier.npy",train_num_fichier)
# np.save("./data/test_num_fichier.npy",test_num_fichier)
# np.save("./data/train_num_activite.npy",train_num_activite)
# np.save("./data/test_num_activite.npy",test_num_activite)

train_squel = np.load("./data/train_squel.npy",allow_pickle=True)
test_squel = np.load("./data/test_squel.npy",allow_pickle=True)
train_num_fichier = np.load("./data/train_num_fichier.npy",allow_pickle=True)
test_num_fichier = np.load("./data/test_num_fichier.npy",allow_pickle=True)
train_num_activite = np.load("./data/train_num_activite.npy",allow_pickle=True)
test_num_activite = np.load("./data/test_num_activite.npy",allow_pickle=True)

print(train_num_activite)

for dossier in liste:
    sous_chemin = chemin + dossier + "/label/label.txt"
    derniere_action = ""
    f = open(sous_chemin, 'r')
    liste_lignes = f.readlines()
    nb_lignes = len(liste_lignes)
    dossier = int(dossier)
    if dossier in train_sub:
        index_act = np.where(train_num_activite == dossier)[0][0]
        print(index_act)
        for i in range(nb_lignes):
            if liste_lignes[i].split("\n")[0] in actions:
                derniere_action = liste_lignes[i]
                print(derniere_action)
            elif liste_lignes[i] != "":
                debut = int(liste_lignes[i].split(" ")[0])
                fin = int(liste_lignes[i].split(" ")[1])
                print("Debut " + str(debut) + " Fin " + str(fin))
                if fin-debut < window:
                    debut = debut - (window - (fin - debut))
                cpt = 0
                for j in range(debut,fin):
                    print(j)
                    index_num = np.where(train_num_fichier==j)[0][0]
                    squel = train_squel[index_act][index_num]
                    print("Squelette")
                    print(squel)

    # elif int(dossier) in test_sub:
