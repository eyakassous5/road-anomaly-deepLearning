# 🛣️ Détection d'anomalies routières par segmentation sémantique

Projet de **Deep Learning / Computer Vision** : détecter, **pixel par pixel**, les objets inattendus sur la route (animaux, rochers, cônes, pneus perdus, véhicules) à partir d'une simple photo, puis déployer le meilleur modèle dans une **application web Streamlit**.

> ⚠️ **Prototype académique.** Le modèle est entraîné sur seulement 60 images. Il ne doit pas être utilisé pour des décisions de sécurité routière ou de conduite autonome.


---

## Sommaire

1. [Objectif](#objectif)
2. [Jeu de données](#jeu-de-données)
3. [Méthodologie](#méthodologie)
4. [Résultats](#résultats)
5. [Application Streamlit](#application-streamlit)
6. [Installation et lancement](#installation-et-lancement)
7. [Structure du dépôt](#structure-du-dépôt)
8. [Limites](#limites)
9. [Pistes d'amélioration](#pistes-damélioration)
10. [Technologies](#technologies)
11. [Auteure](#auteure)

---

## Objectif

Pour un véhicule, savoir **où** se trouve un obstacle compte autant que savoir **qu'il existe**. Ce projet traite la détection d'anomalies comme une **segmentation sémantique binaire** : chaque pixel de l'image est classé *normal* (0) ou *anomalie* (1).

Le projet compare plusieurs familles d'approches :

| Approche | Idée |
|---|---|
| **U-Net** (entraîné from scratch) | Baseline sans pré-entraînement |
| **U-Net + ResNet34** (pré-entraîné ImageNet) | Transfer learning |
| **DeepLabV3+ + ResNet34** | Convolutions dilatées (ASPP) pour le contexte multi-échelle |
| **Autoencodeur convolutionnel** | Approche non supervisée : une forte erreur de reconstruction signale une anomalie |

Une expérience annexe de **classification d'image** (animaux / obstacles / autre) est également incluse.

---

## Jeu de données

- **RoadAnomaly** : 60 images de routes (1280 × 720) avec masques de segmentation sémantique (valeur `2` = anomalie).
- Catégories : animaux (31), obstacles (15), cônes (8), pneus perdus (3), véhicules (3).
- **Fort déséquilibre** : seulement **≈ 9,85 %** des pixels sont anormaux. L'accuracy serait donc trompeuse (un modèle qui prédit « tout est normal » aurait ≈ 90 %).
- Prétraitement : redimensionnement en **512 × 288** (images en interpolation bilinéaire, masques en *nearest neighbor* pour ne pas créer de fausses étiquettes), normalisation ImageNet.

> Source et licence du jeu de données : *à compléter.*

### Découpage sans fuite de données

Plusieurs images proviennent de la **même scène** (quasi-doublons). Un découpage aléatoire ferait fuiter des scènes entre entraînement et test. Le découpage utilise donc `StratifiedGroupKFold` :

- **Group** : une scène n'est jamais répartie sur deux ensembles (contrôle : `groups in >1 split: 0`).
- **Stratified** : les catégories restent proportionnelles.
- Taille : **34 images en train / 14 en validation / 12 en test**.

---

## Méthodologie

1. **Exploration** : visualisation des masques, répartition des catégories, taille des anomalies.
2. **Augmentation** (Albumentations, train uniquement) : retournement horizontal, transformation affine (zoom, translation, rotation ±10°), luminosité/contraste, teinte/saturation. Les transformations géométriques sont appliquées à l'image **et** au masque.
3. **Modèles de segmentation** : U-Net écrite en PyTorch, puis U-Net et DeepLabV3+ via `segmentation-models-pytorch` avec encodeur ResNet34 pré-entraîné.
4. **Perte** : `BCE + Dice` pour compenser le déséquilibre de classes.
5. **Entraînement** : AdamW, scheduler en cosinus, *learning rates différentiels* (encodeur pré-entraîné plus petit que le décodeur), sauvegarde du meilleur checkpoint sur la validation.
6. **Évaluation** : Dice, IoU, précision, rappel ; seuil de décision 0,5 ; **validation croisée à 5 plis** pour estimer la variance.
7. **Autoencodeur convolutionnel** : entraîné à reconstruire les pixels *normaux* (perte MSE masquée) ; la carte d'erreur de reconstruction sert de score d'anomalie (AUROC + seuil optimisé sur la validation).
8. **Analyse d'erreurs** : Dice par image, visualisation des contours prédits.

---

## Résultats

### Jeu de test (12 images, découpage unique)

| Modèle | Précision | Rappel | IoU | **Dice** |
|---|---|---|---|---|
| U-Net (from scratch) | 0,671 | 0,524 | 0,417 | 0,589 |
| U-Net + ResNet34 | 0,708 | 0,623 | 0,496 | 0,663 |
| **DeepLabV3+ + ResNet34** | **0,930** | 0,648 | **0,618** | **0,764** |
| Autoencodeur (erreur de reconstruction) | 0,121 | 0,698 | 0,115 | 0,207 |

### Validation croisée à 5 plis (test de chaque pli)

| Modèle | Dice (moyenne ± écart-type) | IoU (moyenne ± écart-type) |
|---|---|---|
| **DeepLabV3+ + ResNet34** | **0,798 ± 0,061** | 0,668 ± 0,088 |
| U-Net + ResNet34 | 0,789 ± 0,094 | 0,659 ± 0,126 |

Avec 12 images de test, un seul découpage est très bruité (le Dice de U-Net + ResNet34 varie de 0,65 à 0,90 selon le pli). **La validation croisée est la mesure à retenir** : les deux architectures pré-entraînées sont équivalentes.

### Ce qu'il faut retenir

- **Le transfer learning est le facteur n°1** : passer d'un U-Net entraîné from scratch à un encodeur ResNet34 pré-entraîné améliore nettement les résultats avec très peu de données.
- **L'autoencodeur échoue** (AUROC ≈ 0,56 en validation, 0,66 en test ; Dice 0,21, à peine au-dessus d'un modèle qui marquerait tout en anomalie). L'erreur de reconstruction mesure surtout la complexité visuelle de l'image, pas son caractère « anormal ».
- **Les modèles sont plutôt précis mais prudents** : ils ratent davantage d'anomalies (rappel ≈ 0,75) qu'ils n'en inventent (précision ≈ 0,87).
- **Limite principale** : les scènes jamais vues à l'entraînement (champs de rochers, hérisson tchèque) sont mal segmentées.

### Expérience annexe : classification (animaux / obstacles / autre)

Validation croisée à 5 plis, macro-F1 :

| Modèle | Accuracy | Macro-F1 |
|---|---|---|
| ResNet34 pré-entraîné | 0,784 | 0,701 |
| Petit CNN (from scratch) | 0,435 | 0,381 |
| *Baseline « classe majoritaire »* | *0,529* | n/a |

ResNet34 est meilleur dans 5 plis sur 5 ; le petit CNN fait moins bien que la baseline.

---

## Application Streamlit

L'application charge le modèle **DeepLabV3+ / ResNet34** entraîné et permet de :

- téléverser une image de route (PNG, JPG, JPEG, WEBP) ;
- obtenir le **masque d'anomalie prédit** et un **overlay** (anomalies en rouge) ;
- voir le nombre de pixels détectés et le pourcentage de l'image signalé ;
- **télécharger** le masque et l'overlay en PNG.

**Pipeline d'inférence** (identique à l'entraînement) : redimensionnement 512 × 288 → normalisation ImageNet → réseau → sigmoïde → seuil à 0,5.

> Le pourcentage affiché est la fraction de pixels signalés, **pas** une probabilité de danger.

---

## Installation et lancement

**Prérequis** : Python 3.10 ou 3.11, environ 1 Go d'espace disque.

### 1. Récupérer le projet

```bash
git clone https://github.com/<votre-utilisateur>/<nom-du-depot>.git
cd <nom-du-depot>
```

### 2. Placer les poids du modèle

Téléchargez `deeplabv3p_resnet34.pt` (≈ 90 Mo) et placez-le **à côté de `app.py`** :

> 📦 Lien de téléchargement des poids : 

### 3. Créer l'environnement et installer les dépendances

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur ; sinon, allez sur `http://localhost:8501`.

### En cas d'erreur de chargement du modèle

Si `load_state_dict` signale des clés manquantes ou inattendues, **ne modifiez pas le checkpoint** : alignez les versions de `segmentation-models-pytorch` et `timm` avec celles de l'entraînement (voir `requirements.txt`).

---

## Structure du dépôt

```text
.
├── app.py                      # Application Streamlit
├── requirements.txt            # Dépendances de l'application
├── README.md
├── notebooks/
│   └── road_anomaly_segmentation.ipynb   # Entraînement et évaluation (Google Colab, GPU)
├── assets/                     # Captures d'écran (optionnel)
└── deeplabv3p_resnet34.pt      # Poids du modèle (non versionné, voir ci-dessus)
```

Pour rejouer l'entraînement, ouvrez le notebook dans Google Colab avec un GPU et adaptez les chemins Google Drive du jeu de données.

---

## Limites

- **Très peu de données** : 60 images, dont 34 pour l'entraînement et seulement 12 pour le test. Les scores sont donc sensibles au découpage.
- **Scènes jamais vues** : un type d'objet absent de l'entraînement peut passer inaperçu (problème « open-set »).
- **Seuil fixe à 0,5**, non calibré : les sorties ne sont pas des probabilités de danger.
- Résultats valables sur ce jeu de données ; non testé sur d'autres caméras, météos ou pays.

---

## Pistes d'amélioration

- Plus de données (jeux de données publics d'anomalies routières, données synthétiques par copier-coller d'objets).
- **Ensemble** U-Net + DeepLabV3+ (moyenne des probabilités) et *test-time augmentation*.
- Calibration du seuil de décision par catégorie d'anomalie.
- Comparaison dans l'espace des caractéristiques d'un réseau pré-entraîné plutôt qu'en pixels (pour l'approche non supervisée).
- Conteneurisation (Docker) et suivi des expériences (MLflow).

---

## Technologies

Python · PyTorch · segmentation-models-pytorch · timm · Albumentations · scikit-learn · pandas · NumPy · Matplotlib · Pillow · Streamlit · Google Colab

---

## Auteure

**Eya Kassous** : étudiante ingénieure ERP & Business Intelligence, ESPRIT

- LinkedIn : [Eya Kassous](https://www.linkedin.com/in/eya-kassous-713b29216/)
- GitHub : [eyakassous5](https://github.com/eyakassous5)
