import numpy as np
import threading
import queue
import json
import tensorflow as tf
from activity_analyzer.analyzer import ActivityAnalyzer

### ACTIVITES
activites = [0,1,10,31,32,33,38,42,46,58]

path_to_activite = "../dataset/oad_whenwill/npy/pre/"

class Predicter:
    """
        Class in charge of converting frames in image(s) and make predictions.
    """

    def __init__(self, mode,modele_action,taille,ensemble,choix_nb_actions,encodage,modele_activite,modele_prediction, choix_nb_obs, choix_nb_pre, choix_rho):
        """
            Initialize the predicter

            Load the pretrained model, parse an example bvh file to obtain the skeleton
            Get the joints list and the joints list of ignored joints
        """
        self.encodage = encodage
        self.rho = float(choix_rho)
        if encodage == "Base" or encodage == "Unitaire":
            self.nb_lig = 25
            self.nb_col = int(taille)
        elif encodage == "Fusion":
            self.nb_lig = 50
            self.nb_col = int(taille)
        if encodage == "Base" or encodage == "Unitaire":
            self.nb_lig = 25
            self.nb_col = int(taille)
        elif encodage == "Fusion":
            self.nb_lig = 50
            self.nb_col = int(taille)
        if ensemble == 'OAD':
            if choix_nb_actions == '9':
                self.classes = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing',
                                'wiping', 'opening microwave oven', 'Throwing trash']
            elif choix_nb_actions == '11':
                self.classes = ['no action', 'sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating',
                                'writing', 'wiping', 'drinking', 'opening microwave oven', 'Throwing trash']
            else:
                ## EN
                # self.classes = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping', 'drinking', 'opening microwave oven', 'Throwing trash']
                ## FR
                self.classes = ['balayer', 'se gargariser', 'ouvrir le placard', 'se laver les mains', 'manger',
                                'écrire', 'essuyer', 'boire', 'ouvrir le four à micro-ondes', 'Jeter des ordures']

        elif ensemble == 'NTU':
            self.classes = ['drink water', 'sit down', 'stand up', 'clapping', 'point to something', 'check watch',
                            'salute']
        elif ensemble == 'NTU/OAD':
            if choix_nb_actions == '18':
                self.classes = ['no action', 'sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating',
                                'writing', 'wiping', 'drinking',
                                'opening microwave oven', 'Throwing trash', 'drink water', 'sit down', 'stand up',
                                'clapping', 'point to something', 'check watch', 'salute']
            else:
                self.classes = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing',
                                'wiping',
                                'opening microwave oven', 'Throwing trash', 'drink water', 'sit down', 'stand up',
                                'clapping', 'point to something', 'check watch', 'salute']
        self.shape_res = (1,self.nb_lig,self.nb_col,3)
        self.frames = []
        self.predictions = []
        self.futur_obs = np.zeros([int(choix_nb_obs),int(choix_nb_actions)], dtype = int)
        self.futur_pred = []
        self.cpt_pred = 0
        self.cpt_futur = 0
        self.max_pred = 5
        self.derniere_rec = None
        self.nb_obs = int(choix_nb_obs)
        self.nb_action = int(choix_nb_actions)
        self.nb_pre = int(choix_nb_pre)
        self.model = tf.keras.models.load_model(modele_action)
        self.model_pred = tf.keras.models.load_model(modele_prediction)
        self.model_activite = tf.keras.models.load_model(modele_activite)
        self.model.summary()
        self.killed = False
        self.threshold = 0.90
        self.mode = mode
        if self.mode != "web":
            activities_json = json.load(open("./data/activities.json", "rb"))
            self.recon_action = queue.Queue()
            self.activity_analyzer = ActivityAnalyzer(self, activities_json, self.recon_action, self.mode)
            self.activity_analyzer.start()
    
    def update_frames(self, frames):
        """
            Save the received frames in a numpy array

            :param frames: received frames from the client thread
            :type frames: list
        """
        if frames!=None:
            self.frames = self.to_nassim(frames,'foot_to_foot')
            return self.frames

    def kill(self):
        """ Break the activity analyzer loop when called """
        if self.mode != "web":
            self.activity_analyzer.killed = True


    def run(self):
        """
            Main function of this class

            Create a thread and start it
        """
        t = threading.Thread(target=self.predict_classes)
        t.daemon=True
        t.start()

    def prep_image(self,body,joints_list):
        current_frame = []
        joints = body.joints
        # Récupération des jointures squelettes
        for joint_index in joints_list:
            current_joint = []
            current_joint.append(joints[joint_index].Position.x)
            current_joint.append(joints[joint_index].Position.y)
            current_joint.append(joints[joint_index].Position.z)
            current_frame.append(np.array(current_joint))
        return joints, current_frame
    # Vérification des erreurs, si il y a des petites erreurs qui s'imiscent entre deux grosses actions alors on les enlève
    # Une petite erreur correspond à un temps inférieur à 5, valeur modulable
    def pred_actions(self,matrice_entree):
        self.predi = np.asarray(self.model_pred(np.asarray(matrice_entree)))
        n_predi = self.predi.copy().reshape(self.nb_pre, self.nb_action)
        if np.argmax(n_predi[0], axis=-1) != np.argmax(n_predi[1], axis=-1):
            n_predi[0][np.argmax(n_predi[1], axis=-1)] += 2
        n_1 = np.argmax(n_predi[0], axis=-1)
        for i in range(1, self.nb_pre-4):
            lab_1 = np.argmax(n_predi[i], axis=-1)
            lab_2 = np.argmax(n_predi[i + 1], axis=-1)
            lab_3 = np.argmax(n_predi[i + 2], axis=-1)
            lab_4 = np.argmax(n_predi[i + 3], axis=-1)
            lab_5 = np.argmax(n_predi[i + 4], axis=-1)
            if lab_5 == n_1:
                n_predi[i][n_1] += 5
                n_predi[i + 1][n_1] += 5
                n_predi[i + 2][n_1] += 5
                n_predi[i + 3][n_1] += 5
                lab_1 = lab_2 = lab_3 = lab_4 = n_1
            if lab_1 != n_1 and lab_1 == lab_2 and lab_1 == lab_3 and lab_1 != lab_4:
                n_predi[i][n_1] += 5
                n_predi[i + 1][n_1] += 5
                n_predi[i + 2][n_1] += 5
                lab_1 = lab_2 = lab_3 = n_1
            if lab_1 != n_1 and lab_1 == lab_2 and lab_1 == lab_3 and lab_1 == lab_4 and lab_4 != lab_5:
                n_predi[i][n_1] += 5
                n_predi[i + 1][n_1] += 5
                n_predi[i + 2][n_1] += 5
                n_predi[i + 3][n_1] += 5
                n_predi[i + 4][n_1] += 5
                lab_1 = lab_2 = lab_3 = lab_4 = lab_5 = n_1
            if lab_1 != n_1 and lab_1 == lab_2 and lab_2!=lab_3:
                n_predi[i][n_1] += 5
                n_predi[i + 1][n_1] += 5
                lab_1 = lab_2 = n_1
            elif n_1 != lab_1 and lab_1 == lab_2 and lab_2!=lab_3 and lab_3==lab_4 and lab_4!=lab_5:
                n_predi[i][n_1] += 5
                n_predi[i + 1][n_1] += 5
                n_predi[i + 2][lab_5] += 5
                n_predi[i + 3][lab_5] += 5
                lab_1 = np.argmax(n_predi[i], axis=-1)
                lab_2 = np.argmax(n_predi[i + 1], axis=-1)
                lab_3 = np.argmax(n_predi[i + 2], axis=-1)
                lab_4 = np.argmax(n_predi[i + 3], axis=-1)
            if lab_2 != lab_1 and lab_2 != lab_3:
                n_predi[i + 1][lab_1] += 5
                lab_2 = lab_1
            if lab_3 != lab_2 and lab_3 != lab_4:
                n_predi[i+2][lab_2] += 5
                lab_3 = lab_2
            if lab_4 != lab_3 and lab_4!= lab_5:
                n_predi[i+3][lab_3] += 5
                lab_4 = lab_3
            n_1 = lab_1
        return n_predi
    # Fonction de reconnaissance d'activité haut niveau
    def rec_activite(self,matrice_entree):
        activite = self.model_activite(matrice_entree)
        max_1, max_2 = self.max(activite)
        liste_action_1 = np.load(path_to_activite + str(max_1) + ".npy")
        l_act_1 = []
        for i in range(len(liste_action_1)):
            l_act_1.append([liste_action_1[i]])
        liste_action_2 = np.load(path_to_activite + str(max_2) + ".npy")
        l_act_2 = []
        for i in range(len(liste_action_2)):
            l_act_2.append([liste_action_2[i]])
        return max_1, max_2, l_act_1, l_act_2

    # On cherche les deux activités les plus probable
    def max(self,matrice):
        if matrice[0][0] > matrice[0][1]:
            max_1 = matrice[0][0]
            id_1 = 0
            max_2 = matrice[0][1]
            id_2 = 1
        else:
            max_1 = matrice[0][1]
            id_1 = 1
            max_2 = matrice[0][0]
            id_2 = 0
        for i in range(2,matrice[0].shape[0]):
            if matrice[0][i] > max_1:
                max_2 = max_1
                id_2 = id_1
                max_1 = matrice[0][i]
                id_1 = i
            elif matrice[0][i] > max_2:
                max_2 = matrice[0][i]
                id_2 = i
        max_1 = activites[id_1]
        max_2 = activites[id_2]
        return max_1, max_2

    def predict_classes(self):
        """
            Convert received frames in images with the specified method and make predictions
        """
        final_images = self.frames
        final_images = np.reshape(final_images, self.shape_res)
        if final_images != []:
            predictions = self.model.predict(final_images)
            if (np.max(predictions[0], axis=-1) < self.threshold):
                self.predictions = []
                return
            self.predictions = predictions[0]
            if self.mode != "web":
                if self.activity_analyzer.started:
                    predicted_class_index = np.argmax(self.predictions, axis=-1)
                    if self.derniere_rec == self.classes[predicted_class_index]:
                        self.cpt_pred += 1
                        self.derniere_rec = self.classes[predicted_class_index]
                        if self.cpt_pred >= self.max_pred:
                            self.recon_action.put(self.derniere_rec)
                    else:
                        self.derniere_rec = self.classes[predicted_class_index]
                        self.cpt_pred = 0
        else:
            self.predictions = []

    # Correspond à l'encodage unitaire, la noramalisation est réalisé par rapport à la position précédente
    def transform_image_ludl_obo(self,image,weights):
        image = np.asarray(image)
        RGB = image
        height = image.shape[1]
        width = image.shape[0]
        X = np.arange(height)
        Y = np.arange(width)
        RGB = np.squeeze(RGB)
        white = np.ones((width, height)) * 255
        for i in range(3):
            RGB[:, :, i] = np.floor(
                255 * (RGB[:, :, i] - np.amin(RGB[:, :, i])) / (np.amax(RGB[:, :, i]) - np.amin(RGB[:, :, i])))
            RGB[:, :, i] = RGB[:, :, i] * weights + (1 - weights) * white
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in X:
            for j in Y:
                img[i, j] = RGB[j, i]
        return img

    # Correspond à l'encodage de base, la normalisation est réalisée avec toutes les positions
    def transform_image_ludl_base(self,image):
        image = np.asarray(image)
        RGB = image
        height = image.shape[1]
        width = image.shape[0]
        X = np.arange(height)
        Y = np.arange(width)
        RGB = np.squeeze(RGB)
        for i in range(3):
            RGB[:, :, i] = np.floor(
                255 * (RGB[:, :, i] - np.amin(RGB[:, :, i])) / (np.amax(RGB[:, :, i]) - np.amin(RGB[:, :, i])))
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in X:
            for j in Y:
                img[i, j] = RGB[j, i]
        return img

    def to_nassim(self,data, type='no_order'):
        data = [self.get_data(x,type) for x in data]
        if self.encodage == 'Fusion':
            weights_obo = self.get_sequence_energy_obo(data)
            image_base = self.transform_image_ludl_base(data)
            image_obo = self.transform_image_ludl_obo(data, weights_obo)
            image = np.concatenate((image_obo, image_base))
        elif self.encodage == 'Base':
            image_base = self.transform_image_ludl_base(data)
            image = image_base
        elif self.encodage == 'Unitaire':
            weights_obo = self.get_sequence_energy_obo(data)
            image_obo = self.transform_image_ludl_obo(data, weights_obo)
            image = image_obo
        return np.asarray(image)

    def get_data(self,datait,type='no_order'):
        if type=='no_order':
            data = np.asarray(datait)
            data = data.reshape((25,3))
        else:
            spine_base = datait[0]
            spine_mid = datait[1]
            neck = datait[2]
            head = datait[3]
            shoulder_left = datait[4]
            elbow_left = datait[5]
            wrist_left = datait[6]
            hand_left = datait[7]
            shoulder_right = datait[8]
            elbow_right = datait[9]
            wrist_right = datait[10]
            hand_right = datait[11]
            hip_left = datait[12]
            knee_left = datait[13]
            ankle_left = datait[14]
            foot_left = datait[15]
            hip_right = datait[16]
            knee_right = datait[17]
            ankle_right = datait[18]
            foot_right = datait[19]
            spine_shoulder = datait[20]
            handtip_left = datait[21]
            thumb_left = datait[22]
            handtip_right = datait[23]
            thumb_right = datait[24]

            if type=='human':
                data=np.stack((head, neck, spine_shoulder, shoulder_left, shoulder_right, elbow_left, elbow_right,
                                                               wrist_left, wrist_right, thumb_left, thumb_right, hand_left, hand_right, handtip_left,
                                                               handtip_right, spine_mid, spine_base, hip_left, hip_right, knee_left, knee_right,
                                                               ankle_left, ankle_right, foot_left, foot_right))
            else :
                data=np.stack((foot_left, ankle_left, knee_left, hip_left, spine_base, handtip_left, thumb_left,
                                        hand_left, wrist_left, elbow_left, shoulder_left
                                                                   ,spine_shoulder,head,neck, shoulder_right,elbow_right,
                                                                        wrist_right, hand_right,thumb_right
                                                                       , handtip_right, spine_mid, hip_right,
                                                                       knee_right, ankle_right,foot_right))
        return data

    def normalize(self,array):
        min_ = np.min(array, 0)
        max_ = np.max(array, 0)
        return (array - min_) / (max_ - min_)

    def get_sequence_energy_obo(self,sequence):
        energy = np.zeros((len(sequence), 25))
        for i in range(len(sequence)):
            for k in range(25):
                if i == 0:
                    energy[i][k] = np.linalg.norm(sequence[i][k] - sequence[i + 1][k])
                elif i == len(sequence) - 1:
                    energy[i][k] = np.linalg.norm(sequence[i][k] - sequence[i - 1][k])
                else:
                    energy[i][k] = (np.linalg.norm(sequence[i][k] - sequence[i + 1][k]) + np.linalg.norm(
                        sequence[i][k] - sequence[i - 1][k])) / 2
        E = self.normalize(energy)
        w = self.rho * E + (1 - self.rho)
        return w