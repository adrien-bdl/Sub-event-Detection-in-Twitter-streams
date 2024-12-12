Structure du dossier : 

    |_ challenge_data
        |_ train_BERT_400
            # 1 df par match avec les embeddings de 400 tweets par minute

    |_ src
        |_ create_BERT_df.py 
        |   # fichier de création des df train_BERT_400, normalement pas besoin de réutiliser, sauf
              pour plus d'échantillons (>400) ou un différent preprocessing des tweets
        |_ DL_models.py
        |   # définition de différents modèles de DL
        |_ DL_utils.py
        |   # fonctions utiles pour l'entraînement et la validation des modèles de DL
        |_ utils.py
            # fonctions utiles pour manipuler les données etc.

    |_ script
        |_ example_script.py
            # exemple de script qui fait de la classification à partir des embeddings BERT

Chacun peut créer ses scripts dans le dossier script pour faire ses tests.
On peut modifier les fichiers src, mais sans modifier le fonctionnement des fonctions existantes.
Faites vous une copie du script exemple et partez de ça, ça sera plus simple.