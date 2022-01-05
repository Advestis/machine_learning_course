## Consignes pour l'examen

Bienvenue à la partie pratique de l'examen pour le cours *"Machine learning pour l'énergétique"*. Nous vous invitons à lire très attentivement (et en entier) le readme ci-présent avant de commencer votre exercice.

Pendant cet expérience, vous aborderez un problème de classification binaire en utilisant un data-set artificiel disponible à travers l'appelle de la fonction [datasets.make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) de la libraire [sklearn](https://scikit-learn.org/stable/) Chaque point du data-set est caractérisé par 2 features. L'image diponible au lien ci-dessous montre comment les 2 classes sont reparties dans l'espace des données:

![Représentation du data-set](dataset.pdf)

Vous construirez un modèle de machine learning capable de classifier les données (à la fois du test et d'apprentissage) dans le 2 classes avec une *accuracy* maximale attendue de ~96%. Comme on à appris pendant le cours théorique, la régression logistique sera votre meilleure amie. Toutefois, du haut de votre pluriannuelle expérience sur les réseaux des nuerons, vous trouverez les poids optimales du modèle avec la technique de la backpropagation.

*Une première question pour vous:* est-ce que c'est possible de modéliser un problème de régression logistique sous la forme d'un réseaux de nuerons ? Si oui, avec quel nom cet architecture est connue  dans la littérature ?

Pour obtenir la précision désirée, vous augmenterez le data-set en ajoutant des features dérivées (étape de feature engineering) :

<img src="https://render.githubusercontent.com/render/math?math=\{x_1, x_2\} \rightarrow \{x_1, x_2, x_1^2, x_2^2\}">

Par conséquence, la taille de vos inputs sera 4 unités.

- <b>Première exercice.</b> Nous vous demandons de parcourir le fichier main.py que vous trouverez dans la branch exam du projet machine_learning_course avec le quel vous avez pu familiariser lors du TD. Vous devrez:

  - définir la variable NN_ARCHITECTURE selon les indications données ci-dessus (vous devez obtenir une régression logistique)

  - répondre aux questions présentes dans les différentes docstrings

- <b>Deuxième exercice.</b> Dans le module functions_numy.py, nous vous invitons à remplacer la fonction de loss bce (binary cross-entropy) par la MSE (mean squared error). En particulier vous devrez ajouter les parties manquantes du code que vous sont demandées aux lignes ?? . Vérifiez donc la performance de votre code: est-ce que c'était une bonne idée cet remplacement ? Pourquoi la régression logistique nécessite effectivement de la bce ?
