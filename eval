import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# --- PARAMÈTRES DU PROJET ---
CHEMIN_DATASET = 'dataset'  # Assure-toi que ton dossier s'appelle ainsi
DIMS_IMAGE = (150, 150)
TAILLE_BATCH = 32
EPOQUES = 10  # Demandé dans la Partie 4 [cite: 61]

print("--- INITIALISATION ---")

# ==========================================
# PARTIE 1 & 2 : PRÉPARATION (Via ImageDataGenerator)
# Consigne 
# ==========================================

# On utilise ImageDataGenerator pour normaliser et séparer validation/train
# On ajoute aussi un peu d'augmentation de données ici (zoom/horizontal flip) pour la Partie 5
generateur_donnees = ImageDataGenerator(
    rescale=1./255,           # Normalisation [0,1]
    validation_split=0.2,     # 20% pour la validation [cite: 37]
    horizontal_flip=True,     # Amélioration simple
    zoom_range=0.2            # Amélioration simple
)

print("Chargement des images d'entraînement...")
donnees_entrainement = generateur_donnees.flow_from_directory(
    CHEMIN_DATASET,
    target_size=DIMS_IMAGE,
    batch_size=TAILLE_BATCH,
    class_mode='binary',
    subset='training',
    shuffle=True
)

print("Chargement des images de validation...")
donnees_validation = generateur_donnees.flow_from_directory(
    CHEMIN_DATASET,
    target_size=DIMS_IMAGE,
    batch_size=TAILLE_BATCH,
    class_mode='binary',
    subset='validation'
)

# Petit check des classes
labels_map = (donnees_entrainement.class_indices)
print(f"Classes détectées : {labels_map}")

# ==========================================
# PARTIE 3 : CONSTRUCTION DU CNN
# Architecture demandée [cite: 46, 47, 48, 49]
# ==========================================

modele = Sequential()

# 1ère couche de convolution
modele.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
modele.add(MaxPooling2D(2, 2))

# 2ème couche de convolution
modele.add(Conv2D(64, (3, 3), activation='relu'))
modele.add(MaxPooling2D(2, 2))

# Aplatissement
modele.add(Flatten())

# Couches Dense
modele.add(Dense(128, activation='relu'))

# AMÉLIORATION PARTIE 5: Ajout de Dropout pour éviter le surapprentissage
modele.add(Dropout(0.5))

# Sortie (Sigmoid car binaire)
modele.add(Dense(1, activation='sigmoid'))

# Compilation [cite: 50, 51, 52]
modele.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

modele.summary()

# ==========================================
# PARTIE 4 : ENTRAÎNEMENT
# ==========================================

print("\n--- DÉBUT DE L'ENTRAÎNEMENT ---")
historique = modele.fit(
    donnees_entrainement,
    steps_per_epoch=donnees_entrainement.samples // TAILLE_BATCH,
    validation_data=donnees_validation,
    validation_steps=donnees_validation.samples // TAILLE_BATCH,
    epochs=EPOQUES
)

# ==========================================
# VISUALISATION DES COURBES [cite: 62, 63, 64]
# ==========================================

def afficher_courbes(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))

    # Graphique Précision
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Val Accuracy', color='orange')
    plt.title('Précision (Accuracy)')
    plt.legend()
    plt.grid(True)

    # Graphique Perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Val Loss', color='orange')
    plt.title('Perte (Loss)')
    plt.legend()
    plt.grid(True)
    
    plt.show()

afficher_courbes(historique)

# ==========================================
# PARTIE 6 : TEST SUR NOUVELLES IMAGES [cite: 85]
# ==========================================

def tester_image(chemin_fichier):
    if not os.path.exists(chemin_fichier):
        print(f"Erreur : L'image {chemin_fichier} n'existe pas.")
        return

    # Chargement et prétraitement manuel comme demandé [cite: 87]
    img = image.load_img(chemin_fichier, target_size=DIMS_IMAGE)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0  # Important: Normalisation

    prediction = modele.predict(img_tensor)
    score = prediction[0][0]

    # Interprétation
    # Attention: vérifie tes dossiers, souvent 0=chat, 1=chien
    if score > 0.5:
        resultat = "CHIEN"
        prob = score
    else:
        resultat = "CHAT"
        prob = 1 - score
        
    print(f"Résultat pour {os.path.basename(chemin_fichier)} : {resultat} ({prob:.2%} de certitude)")
    
    # Affichage rapide
    plt.imshow(img)
    plt.title(f"{resultat} ({prob:.1%})")
    plt.axis('off')
    plt.show()

print("\n--- TEST SUR IMAGES EXTERNES ---")
# Remplace par tes propres noms d'images
# tester_image('test_images/mon_chat.jpg')
# tester_image('test_images/mon_chien.jpg')
