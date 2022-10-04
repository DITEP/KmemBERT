# READ ME - Visualisation of patient interpretation on CLB data
# Auteur de ce script : Théo Di Piazza (Centre Léon Bérard), avec l'aide des scripts de Mohamed Aymen Qabel.


Le fichier interpretation_patient_CLB_script.ipynb est un exemple de visualisation d'interprétation patient, utilisé dans le papier (sur données CLB).

Pour exécuter le script :
- Si besoin, un jeu de données (format .csv) avec les colonnes suivantes : Noigr, Texte, Date cr, Date deces (si vous souhaitez utiliser des vrais CR issus d'une database). Pas nécessaire si vous entrez les données textuelles manuellement.
- Du dossier kmembert-base associé au modèle KmemBERT Base.
- Du fichier json (large.json) contenant tous les mots du vocabulaire médical, disponible sur le GitHub du projet.
- De la librairie 'shap' dans votre environnement python.

Recommandation : 
- Placer ce script 'interpretation_patient_CLB_script.ipynb' dans le dossier KmemBERT. Pas nécessairement dans KMEMBERT/kmembert/Visualization_CLB.



