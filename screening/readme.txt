4 dossiers :

1 - testT2 : 
dossier avec l'ensemble des scripts sur la création du jeu de données jusqu'au test et interprétations.

2 - Results : 
Résultats du test du T2. Scores et métriques pour chaque cohorte.

3 - Training Essais Cliniques : 
Scripts pour le FineTuning de T2 sur le screening (classification).
Pour entrainer le modèle, penser à ajouter kmembert-T2 et kmembert-BASE, et les données dans "data\\ehr".
La sortie du modèle est une classification (0 ou 1). Pour modifier l'architecture, aller dans "kmembert\\transformer_aggregator.py".
La données en input doit contenir les colonnes suivante : date d'entrée, de dernière nouvelle (format 01022022 pour 1 Février 2022), le Noigr (code du patient) et le texte du compte-rendu.
Egalement les colonnes suivantes pour analyser les résultats post-entrainement : 
MyIndice (int) : qui permet d'associer chaque prédiction à un indice unique.
FLAG_most_Recent (0 ou 1) : qui permet d'indiquer quel est le compte-rendu le plus récent pour chaque patient (pour pouvoir réaliser une prédiction).


4 - Data Augmentation : 
Scripts pour EDA français. v1, pas définitive. Attention si vous l'utilisez.

FIN.
