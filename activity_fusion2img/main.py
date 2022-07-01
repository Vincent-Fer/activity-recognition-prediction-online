from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import time
import cv2
import PySimpleGUI as sg
import numpy as np
from pathlib import Path
import sys
import queue
import threading
from predicter import Predicter
import argparse

joints_list = [
    PyKinectV2.JointType_SpineBase,
    PyKinectV2.JointType_SpineMid,
    PyKinectV2.JointType_Neck,
    PyKinectV2.JointType_Head,
    PyKinectV2.JointType_ShoulderLeft,
    PyKinectV2.JointType_ElbowLeft,
    PyKinectV2.JointType_WristLeft,
    PyKinectV2.JointType_HandLeft,
    PyKinectV2.JointType_ShoulderRight,
    PyKinectV2.JointType_ElbowRight,
    PyKinectV2.JointType_WristRight,
    PyKinectV2.JointType_HandRight,
    PyKinectV2.JointType_HipLeft,
    PyKinectV2.JointType_KneeLeft,
    PyKinectV2.JointType_AnkleLeft,
    PyKinectV2.JointType_FootLeft,
    PyKinectV2.JointType_HipRight,
    PyKinectV2.JointType_KneeRight,
    PyKinectV2.JointType_AnkleRight,
    PyKinectV2.JointType_FootRight,
    PyKinectV2.JointType_SpineShoulder,
    PyKinectV2.JointType_HandTipLeft,
    PyKinectV2.JointType_ThumbLeft,
    PyKinectV2.JointType_HandTipRight,
    PyKinectV2.JointType_ThumbRight,
]

class KinectListener(threading.Thread):
    """
        This class represents the listener we implement for the Kinect camera.
        It receives frames from the Kinect (in real-time) and gets skeleton data from them. 
    """

    def __init__(self,mode,modele_action, taille, encodage, ensemble, cooldown, choix_nb_actions, modele_activite, choix_nb_obs_activite, choix_nb_activite, modele_prediction, choix_nb_obs, choix_nb_pre, choix_rho):
        """
            Initialize the listener :
            - Connect to the Kinect camera with the pykinect2 library
            - Create the predicter and make a first prediction to load the model

            :param mode: Mode we want to use
            :type mode: string
        """
        threading.Thread.__init__(self)
        if encodage=="Base" or encodage=="Unitaire":
            self.nb_lig = 25
            self.nb_col = int(taille)
        elif encodage=="Fusion":
            self.nb_lig = 50
            self.nb_col = int(taille)
        if ensemble == 'OAD':
            if choix_nb_actions=='9':
                self.classes = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping', 'opening microwave oven', 'Throwing trash']
            elif choix_nb_actions=='11':
                self.classes = ['no action', 'sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping', 'drinking', 'opening microwave oven', 'Throwing trash']
            else:
                ## EN
                # self.classes = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping', 'drinking', 'opening microwave oven', 'Throwing trash']
                ## FR
                self.classes = ['balayer', 'se gargariser', 'ouvrir le placard', 'se laver les mains', 'manger', 'écrire', 'essuyer', 'boire', 'ouvrir le four à micro-ondes', 'Jeter des ordures']

        elif ensemble == 'NTU':
            self.classes = ['drink water', 'sit down', 'stand up', 'clapping', 'point to something', 'check watch', 'salute']
        elif ensemble == 'NTU/OAD':
            if choix_nb_actions == '18':
                self.classes = ['no action', 'sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping', 'drinking',
                               'opening microwave oven', 'Throwing trash','drink water','sit down','stand up','clapping','point to something','check watch','salute']
            else:
                self.classes = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping',
                               'opening microwave oven', 'Throwing trash','drink water','sit down','stand up','clapping','point to something','check watch','salute']

        self.mode = mode
        self.ensemble = ensemble
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_BodyIndex)
        # Processus en parallèle de prédiction d'action et reconnaisance d'action/activité
        self.predicting_thread = Predicter(self.mode,modele_action,taille,ensemble,choix_nb_actions,encodage,modele_activite,modele_prediction,choix_nb_obs, choix_nb_pre, choix_rho)
        self.killed = False
        self.pret_predir = False
        # Tableaux
        self.predi = []
        self.valide = []
        self.obs = []
        self.matrice_entree = []
        # Tableau pour les couleurs vertes ou rouges
        self.tab_color = []
        self.tab_color_f1 = []
        self.tab_color_f2 = []
        # Nombre d'observaiton, d'action et de prédiction définis dans IHM configuration
        self.nb_obs = int(choix_nb_obs)
        self.nb_action = int(choix_nb_actions)
        self.nb_pre = int(choix_nb_pre)
        for i in range(100):
            self.valide.append("X")
        # Compteurs
        self.cpt_no_action = 0
        self.cpt_obs = 0
        self.cpt_pred = 0
        self.cpt_derive_1 = 0
        self.cpt_derive_2 = 0
        # Attente entre image
        self.cooldown = int(cooldown)
        self.cooldown_max = int(cooldown)
        # Liste de position squelette
        self.frames = queue.Queue()
        # Init IHM utilisation
        self.image_viewer_column = None
        self.rgb_column = None
        self.rec_line = None
        self.activite1_line = None
        self.activite2_line = None
        self.message = None
        self.liste_obs = None
        self.liste_futur = None
        self.liste_futur_fichier_1 = None
        self.liste_futur_fichier_2 = None
        self.temps_rec = None
        self.temps_pred = None
        self.layout = None
        self.window = None
        self.initialisation_ihm()
    #
    def draw_body_bone(self, img, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState
        joint1State = joints[joint1].TrackingState

        # Les deux jointures fournis en paramètres ne sont pas trackés
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked):
            return img

        # Les 2 jointures fournis en paramètres sont supposées
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return img

        # au moins une jointure est ok
        if jointPoints[joint0].x in [float("-inf"),float("inf")]: return img
        if jointPoints[joint1].x in [float("-inf"), float("inf")]: return img
        start = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
        end = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))

        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.line(img, start, end, color=(255,0,0), thickness=8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except:
            pass

    def draw_body(self,img, joints, jointPoints, color):
        # Torse
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_Head,
                            PyKinectV2.JointType_Neck)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_Neck,
                            PyKinectV2.JointType_SpineShoulder)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_SpineMid)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineMid,
                            PyKinectV2.JointType_SpineBase)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderLeft)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineBase,
                            PyKinectV2.JointType_HipRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_SpineBase,
                            PyKinectV2.JointType_HipLeft)
        # Bras droit
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight,
                            PyKinectV2.JointType_ElbowRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ElbowRight,
                            PyKinectV2.JointType_WristRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_HandRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HandRight,
                            PyKinectV2.JointType_HandTipRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_ThumbRight)
        # Bras gauche
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft,
                            PyKinectV2.JointType_ElbowLeft)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft,
                            PyKinectV2.JointType_WristLeft)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristLeft,
                            PyKinectV2.JointType_HandLeft)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HandLeft,
                            PyKinectV2.JointType_HandTipLeft)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_WristLeft,
                            PyKinectV2.JointType_ThumbLeft)
        # Jambe droite
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HipRight,
                            PyKinectV2.JointType_KneeRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_KneeRight,
                            PyKinectV2.JointType_AnkleRight)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_AnkleRight,
                            PyKinectV2.JointType_FootRight)
        # Jambe gauche
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_HipLeft,
                            PyKinectV2.JointType_KneeLeft)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_KneeLeft,
                            PyKinectV2.JointType_AnkleLeft)
        img = self.draw_body_bone(img, joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft,
                            PyKinectV2.JointType_FootLeft)
        return img
    def run(self):
        global liste_action_1, liste_action_2, color_frame2
        self.window = sg.Window("Image", self.layout).finalize()
        try:
            while not self.killed:
                event, values = self.window.read(timeout=0)
                if event == sg.WIN_CLOSED:
                    break
                # si une image rgb est récupérée de la kinect
                if self.kinect.has_new_color_frame():
                    color_frame2 = self.kinect.get_last_color_frame().reshape(((1080,1920,4))).astype(np.uint8)
                # si la kinect repère un corps
                if self.kinect.has_new_body_frame():
                    debut_rec = time.time()
                    body_frame = self.kinect.get_last_body_frame()
                    if body_frame is not None:
                        for i in range(0, self.kinect.max_body_count):
                            body = body_frame.bodies[i]
                            if not body.is_tracked:
                                continue
                            joints, current_frame = self.predicting_thread.prep_image(body,joints_list)
                            # Envoie des 25 jointures à la fonction locale update frames
                            img = self.update_frames(current_frame)
                            # si un corps est repéré
                            if self.kinect.has_new_body_index_frame():
                                # on récupère les positions des jointures par rapport à l'image
                                joint_points = self.kinect.body_joints_to_color_space(joints)
                                # on dessine le squelette sur l'image rgb
                                new_img = self.draw_body(color_frame2, joints, joint_points, color=(0,0,255))
                                new_img = cv2.resize(new_img, (720, 480))
                                new_img = cv2.imencode('.png', new_img)[1].tobytes()
                                # on affiche l'image avec squelette dans la zone image rgb de la fenêtre
                                self.window["-image_rgb-"].update(data=new_img)
                                self.window.refresh()
                            if type(img)!=type(None):
                                # Affichage de l'image encodée
                                img = cv2.resize(img, (img.shape[1]*10, img.shape[0]*10))
                                img = cv2.imencode('.png', img)[1].tobytes()
                                self.window["-image_encode-"].update(data=img)
                                # Récupération de la reconnaissance
                                rec = kinectListener.predicting_thread.predictions
                                if len(rec)>0:
                                    self.cpt_no_action = 0
                                    # Affichage de la reconnaissance d'action
                                    self.window["-label_action_rec-"].update(self.classes[np.argmax(rec,axis=-1)])
                                    fin_rec = time.time()
                                    temps_rec = fin_rec - debut_rec
                                    self.window["-temps_rec-"].update("{:.2f}".format(temps_rec*1000))
                                    if self.ensemble=='OAD':
                                        # Si le nombre d'observation est en dessous du nombre définis dans la configuration dans l'IHM
                                        if len(self.obs)<self.nb_obs:
                                            sous_tab = np.zeros((self.nb_action,1))
                                            sous_tab[np.argmax(rec,axis=-1)] = 1
                                            if not self.pret_predir:
                                                # on rajoute une valeur à la fin de la matrice
                                                self.matrice_entree.append(sous_tab)
                                            else:
                                                # on enlève la première valeur de la matrice pour en rajouter une à la dernière place
                                                self.matrice_entree = np.concatenate((self.matrice_entree[1:self.nb_obs],[sous_tab]))
                                            self.obs.append([self.classes[np.argmax(rec, axis=-1)]])
                                            self.cpt_obs += 1
                                        elif len(self.obs) == self.nb_obs:
                                            debut_pred = time.time()
                                            n_matrice_entree = np.asarray(self.matrice_entree).reshape((-1, self.nb_obs, self.nb_action, 1))
                                            self.predi = kinectListener.predicting_thread.pred_actions(n_matrice_entree)
                                            for i in range(self.nb_pre):
                                                act_p = [self.classes[np.argmax(self.predi[i], axis=-1)]]
                                                self.valide[i] = act_p
                                            max_1, max_2, liste_action_1, liste_action_2 = kinectListener.predicting_thread.rec_activite(n_matrice_entree)
                                            self.window['-label_activite1_rec-'].update(max_1)
                                            self.window['-label_activite2_rec-'].update(max_2)
                                            self.window['-list_action_pred_fichier_1-'].update(liste_action_1)
                                            self.window['-list_action_pred_fichier_2-'].update(liste_action_2)
                                            self.window['-list_action_pred-'].update(values=self.valide)
                                            fin_pred = time.time()
                                            temps_pred = fin_pred - debut_pred
                                            self.window['-temps_pred-'].update("{:.2f}".format(temps_pred*1000))
                                            self.obs = []
                                            self.predi = []
                                            self.cpt_obs = 0
                                            self.pret_predir = True
                                        self.window['-list_action_obs-'].update(values=self.obs)
                                        # si une prédiction est déjà réalisée pour vérifier par rapport aux nouvelles reconnaissances
                                        if self.pret_predir == True and self.valide[0] not in ["X"]:
                                            # si la prédiction à un temps t est égale à la dernière reconnaissance au temps t alors la prédiction est bonne
                                            # Pour pred de farha (premiere liste pred)
                                            if self.valide[self.cpt_obs][0] == self.classes[np.argmax(rec,axis=-1)]:
                                                self.tab_color.append((self.cpt_obs,"green"))
                                            else:
                                                self.tab_color.append((self.cpt_obs,"red"))
                                            # si la prédiction à un temps t est égale à la dernière reconnaissance au temps t alors la prédiction est bonne
                                            # Pour pred du fichier 1 (deuxieme liste pred)
                                            if liste_action_1[self.cpt_obs][0] == self.classes[np.argmax(rec,axis=-1)]:
                                                self.tab_color_f1.append((self.cpt_obs, "green"))
                                                self.cpt_derive_1 = 0
                                            else:
                                                self.tab_color_f1.append((self.cpt_obs, "red"))
                                                self.cpt_derive_1 += 1
                                            # si la prédiction à un temps t est égale à la dernière reconnaissance au temps t alors la prédiction est bonne
                                            # Pour pred du fichier 2 (deuxieme liste pred)
                                            if liste_action_2[self.cpt_obs][0] == self.classes[np.argmax(rec,axis=-1)]:
                                                self.tab_color_f2.append((self.cpt_obs, "green"))
                                                self.cpt_derive_2 = 0
                                            else:
                                                self.tab_color_f2.append((self.cpt_obs, "red"))
                                                self.cpt_derive_2 += 1
                                            # Si le nombre d'erreur dans le fichier 2 est inférieur au fichier 1 alors on switch activité 1 et 2
                                            if self.cpt_derive_1 > 10 and self.cpt_derive_2 < 10:
                                                self.window['-message-'].update('Pas activité 1, activité 2 ?')
                                                attente_color = self.tab_color_f1
                                                attente = liste_action_1
                                                attente_max = max_1
                                                liste_action_1 = liste_action_2
                                                self.tab_color_f1 = self.tab_color_f2
                                                liste_action_2 = attente
                                                self.tab_color_f2 = attente_color
                                                max_1 = max_2
                                                max_2 = attente_max
                                                self.window['-label_activite1_rec-'].update(max_1)
                                                self.window['-label_activite2_rec-'].update(max_2)
                                                self.cpt_derive_1 = 0
                                                self.cpt_derive_2 = 0
                                            # Si les deux prédictions pour les fichiers 1 et 2 ont plus de 10 erreurs d'affilés alors on est dans aucune des deux acitivités
                                            elif self.cpt_derive_1 > 10 and self.cpt_derive_2 > 10:
                                                self.pret_predir = False
                                                self.window['-message-'].update('Aucune activité, nouvelle prédiction')
                                                n_matrice_entree = np.asarray(self.matrice_entree).reshape(-1, self.nb_obs, self.nb_action, 1)
                                                self.predi = kinectListener.predicting_thread.pred_actions(n_matrice_entree)
                                                for i in range(self.nb_pre):
                                                    act_p = [self.classes[np.argmax(self.predi[i], axis=-1)]]
                                                    self.valide[i] = act_p
                                                max_1, max_2, liste_action_1, liste_action_2 = kinectListener.predicting_thread.rec_activite(n_matrice_entree)
                                                self.window['-label_activite1_rec-'].update(max_1)
                                                self.window['-label_activite2_rec-'].update(max_2)
                                                self.obs = []
                                                self.tab_color = []
                                                self.tab_color_f1 = []
                                                self.tab_color_f2 = []
                                                self.predi = []
                                                self.cpt_obs = 0
                                                self.cpt_derive_1 = 0
                                                self.cpt_derive_2 = 0
                                                self.pret_predir = True
                                            self.window['-list_action_obs-'].update(values=self.obs)
                                            self.window['-list_action_pred-'].update(values=self.valide,row_colors=self.tab_color)
                                            self.window['-list_action_pred_fichier_1-'].update(values=liste_action_1,row_colors=self.tab_color_f1)
                                            self.window['-list_action_pred_fichier_2-'].update(values=liste_action_2,row_colors=self.tab_color_f2)
                                else:
                                    # on considère un 'No action' si aucune action n'est trouvé sur 3 reconnaissances/frames
                                    if self.cpt_no_action >= 3:
                                        self.window["-label_action_rec-"].update("No action")
                                    self.cpt_no_action += 1
                                self.window.refresh()

            self.kinect.close()
            self.kill()
            kinectListener.kill()
            self.window.close()
        except KeyboardInterrupt:
            self.kinect.close()

    def kill(self):
        """
            Kill the predicter and the listener
        """
        self.killed = True
        self.predicting_thread.kill()

    def update_frames(self, frame):
        """
            Update the list of frames we have. When complete and cooldown=0, send the frames to the predicter.
        """
        if self.frames.qsize() == self.nb_col:
            self.frames.get()
            self.frames.put(np.array(frame))
            if self.cooldown == 0:
                img = self.predicting_thread.update_frames(list(self.frames.queue))
                self.predicting_thread.run()
                self.cooldown = self.cooldown_max
                return img
            else:
                self.cooldown -= 1
        else:
            self.frames.put(np.array(frame))

    def initialisation_ihm(self):
        self.image_viewer_column = [
            [sg.Image(key="-image_encode-")]
        ]
        self.rgb_column = [
            [sg.Image(key="-image_rgb-")]
        ]
        self.rec_line = [
            [sg.Text("Action reconnue : ", font=("Arial", 20)),
             sg.Text(key="-label_action_rec-", font=("Arial", 25), text_color='white')],
        ]
        self.activite1_line = [
            [sg.Text("Activité 1 : ", font=("Arial", 20)),
             sg.Text(key="-label_activite1_rec-", font=("Arial", 25), text_color='white')],
        ]
        self.activite2_line = [
            [sg.Text("Activité 2 : ", font=("Arial", 20)),
             sg.Text(key="-label_activite2_rec-", font=("Aria"
                                                        "l", 25), text_color='white')],
        ]
        self.message = [
            [sg.Text("Message : ", font=("Arial", 20)),
             sg.Text(key="-message-", font=("Arial", 25), text_color='white')],
        ]
        self.liste_obs = [
            [sg.Table(values=[], headings=['observation'], key='-list_action_obs-', num_rows=50)]
        ]
        self.liste_futur = [
            [sg.Table(values=[], headings=['prédiction'], key='-list_action_pred-', num_rows=50)]
        ]
        self.liste_futur_fichier_1 = [
            [sg.Table(values=[], headings=['fichier_1'], key='-list_action_pred_fichier_1-', num_rows=50)]
        ]
        self.liste_futur_fichier_2 = [
            [sg.Table(values=[], headings=['fichier_2'], key='-list_action_pred_fichier_2-', num_rows=50)]
        ]
        self.temps_rec = [
            [sg.Text("Temps reconnaissance : ", font=("Arial", 20)),
             sg.Text(key="-temps_rec-", font=("Arial", 20)),
             sg.Text(" ms", font=("Arial", 20))]
        ]
        self.temps_pred = [
            [sg.Text("Temps prédiction : ", font=("Arial", 20)),
             sg.Text(key="-temps_pred-", font=("Arial", 20)),
             sg.Text(" ms", font=("Arial", 20))]
        ]
        self.layout = [
            [sg.Column(self.image_viewer_column),
             sg.VSeparator(),
             sg.Column(self.rgb_column),
             sg.VSeparator(),
             sg.Column(self.liste_obs),
             sg.Column(self.liste_futur),
             sg.Column(self.liste_futur_fichier_1),
             sg.Column(self.liste_futur_fichier_2)
             ],
            [sg.Column(self.rec_line),
             sg.VSeparator(),
             sg.Column(self.activite1_line),
             sg.VSeparator(),
             sg.Column(self.activite2_line),
             sg.VSeparator(),
             sg.Column(self.message)
             ],
            [sg.Column(self.temps_rec),
             sg.VSeparator(),
             sg.Column(self.temps_pred)]
        ]

def selection_fichier():
    sg.theme("DarkBlue")
    modele_action = None
    taille = None
    encodage = None
    ensemble = None
    cooldown = None
    choix_nb_actions = None
    modele_activite = None
    choix_nb_obs_activite = None
    choix_nb_activite = None
    modele_prediction = None
    choix_nb_obs = None
    choix_nb_pre = None
    choix_rho = None
    quitter = False
    layout = [
        [
            sg.Text('Modèle action :'),
            sg.InputText(key='-modele_action-', default_text='../modeles/fusion_2img/30/93_53_oad_50d.h5'),
            sg.FileBrowse()
        ],
        [
            sg.Text('Nb position :', font='Arial', justification='left'),
            sg.Combo(['40', '30', '20', '10'], default_value='30', key='-choix_taille-'),
            sg.Text('Encodage :', font='Arial', justification='left'),
            sg.Combo(['Unitaire', 'Base', 'Fusion'], default_value='Fusion', key='-choix_encodage-'),
            sg.Text('Ensemble de données :', font='Arial', justification='left'),
            sg.Combo(['OAD', 'NTU', 'NTU/OAD'], default_value='OAD', key='-choix_ensemble-'),
            sg.Text('Cooldown :', font='Arial', justification='left'),
            sg.Combo(['1', '5', '10', '15', '20', '25', '30', '35', '40'], default_value='1', key='-choix_cooldown-'),
            sg.Text('Nb actions :', font='Arial', justification='left'),
            sg.Combo(['7', '8', '9', '10', '11', '16', '18'], default_value='10', key='-choix_nb_actions-'),
            sg.Text('Rho :', font='Arial', justification='left'),
            sg.Combo(['0', '0.5', '1'], default_value='1', key='-choix_rho-'),
        ],
        [
            sg.Text('Modèle activité :'),
            sg.InputText(key='-modele_activite-', default_text='../modeles/pred_farha/activite_100_10.h5'),
            sg.FileBrowse(),
            sg.Text('Nb obs :', font='Arial', justification='left'),
            sg.Combo(['25', '50', '75'], default_value='50', key='-choix_nb_obs_activite-'),
            sg.Text('Nb activité :', font='Arial', justification='left'),
            sg.Combo(['10', '30', '59'], default_value='10', key='-choix_nb_activite-'),
        ],
        [
            sg.Text('Modèle prédiction :'),
            sg.InputText(key='-modele_prediction-', default_text='../modeles/pred_farha/pred_97_8_10.h5'),
            sg.FileBrowse(),
            sg.Text('Nb obs :', font='Arial', justification='left'),
            sg.Combo(['25', '50', '75'], default_value='50', key='-choix_nb_obs-'),
            sg.Text('Nb pre :', font='Arial', justification='left'),
            sg.Combo(['75', '100', '125'], default_value='100', key='-choix_nb_pre-')
        ],
        [sg.Button("Go")],
    ]
    window = sg.Window('Configuration', layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == "Go":
            modele_action = values['-modele_action-']
            taille = values['-choix_taille-']
            encodage = values['-choix_encodage-']
            ensemble = values['-choix_ensemble-']
            cooldown = values['-choix_cooldown-']
            choix_nb_actions = values['-choix_nb_actions-']
            modele_activite = values['-modele_activite-']
            choix_nb_obs_activite = values['-choix_nb_obs_activite-']
            choix_nb_activite = values['-choix_nb_activite-']
            modele_prediction = values['-modele_prediction-']
            choix_nb_obs = values['-choix_nb_obs-']
            choix_nb_pre = values['-choix_nb_pre-']
            choix_rho = values['-choix_rho-']
            while True:
                if not Path(modele_action).is_file():
                    if modele_action == '':
                        sg.popup_ok('Select a file to go !')
                    else:
                        sg.popup_ok('File not exist !')
                    modele_action = sg.popup_get_file("", no_window=True)
                    if modele_action == '':
                        break
                    window['-modele-'].update(modele_action)
                else:
                    quitter = True
                    break
            if quitter:
                break
    window.close()
    return modele_action, taille, encodage, ensemble, cooldown, choix_nb_actions, modele_activite, choix_nb_obs_activite, choix_nb_activite, modele_prediction, choix_nb_obs, choix_nb_pre, choix_rho

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Choose the mode (training or recognition or web)",
        action="store",
        required=True,
        default="web"
    )
    args = parser.parse_args()
    if args.mode != "training" and args.mode != "recognition" and args.mode != "web":
        parser.print_help()
        sys.exit(-1)
    modele_action, taille, encodage, ensemble, cooldown, choix_nb_actions, modele_activite, choix_nb_obs_activite, choix_nb_activite, modele_prediction, choix_nb_obs, choix_nb_pre, choix_rho = selection_fichier()
    kinectListener = KinectListener(args.mode,modele_action, taille, encodage, ensemble, cooldown, choix_nb_actions, modele_activite, choix_nb_obs_activite, choix_nb_activite, modele_prediction, choix_nb_obs, choix_nb_pre, choix_rho)
    kinectListener.run()
    kinectListener.kill()
