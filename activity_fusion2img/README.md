Dossier contenant l'application python pour la reconnaissance d'action, d'activité et l'anticipation d'action.

Des prérequis sont à respecter pour faire fonctionner l'application :

1. Installer le SDK Kinect fournis par Microsoft ([Link](https://www.microsoft.com/en-us/download/details.aspx?id=44561))
2. Installer la librairie python "PyKinect2" via : pip install pykinect2  
3. Régler les erreurs lors de l'execution avec le remplacement de : 
```python
assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)
```
par :
```python
import numpy.distutils.system_info as sysinfo
required_size = 64 + sysinfo.platform_bits / 4
assert sizeof(tagSTATSTG) == required_size, sizeof(tagSTATSTG)
```
4. Modifier tous les time.clock() par time.time() dans le fichier PyKinectRunTime.py (à vérifier dans l'erreur soulevée).

********
Pour lancer l'application il faut utiliser la commande :
  - python main.py --mode recognition
