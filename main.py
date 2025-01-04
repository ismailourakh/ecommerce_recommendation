import pandas as pd
from gensim.models import Word2Vec
import numpy as np


# Fonction pour charger et nettoyer les données du fichier CSV
def load_data(file_path):
    # Charger le fichier CSV
    df = pd.read_csv(file_path, header=None)

    # Filtrer les lignes où toutes les valeurs sont vides
    df = df.dropna(axis=0, how='all')  # Supprimer les lignes complètement vides

    # Nettoyer les valeurs vides dans chaque ligne et convertir en une liste de produits
    df = df.apply(lambda x: x.dropna().tolist(), axis=1)
    return df


# Fonction pour entraîner le modèle Word2Vec
def train_word2vec_model(orders):
    # Entraîner le modèle Word2Vec sur les commandes d'achats
    model = Word2Vec(sentences=orders, vector_size=100, window=5, min_count=1, workers=4)
    model.save("product_word2vec.model")  # Sauvegarder le modèle
    return model


# Fonction pour calculer le centre des vecteurs
def calculate_vector_center(model):
    # Obtenir tous les vecteurs des mots dans le vocabulaire
    vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    # Calculer la moyenne (centre des vecteurs)
    center = np.mean(vectors, axis=0)
    return center


# Fonction pour suggérer des produits proches du centre
def suggest_products_near_center(model, center, topn=5):
    # Calculer les distances de chaque produit au centre
    distances = {}
    for word in model.wv.index_to_key:
        vector = model.wv[word]
        distance = np.linalg.norm(vector - center)  # Distance euclidienne
        distances[word] = distance

    # Trier par proximité au centre (distances les plus petites en premier)
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    # Retourner les produits les plus proches
    return sorted_distances[:topn]


# Fonction principale pour récupérer des produits similaires ou fréquents
def get_similar_products(model, product_name, center, topn=5):
    try:
        # Trouver les mots les plus similaires au produit donné
        similar_products = model.wv.most_similar(product_name, topn=topn)
        print(f"Produits similaires pour '{product_name}':")
        for product, similarity in similar_products:
            print(f"{product}: {similarity}")
    except KeyError:
        print(f"Le produit '{product_name}' n'est pas dans le modèle.")
        print("Produits génériques suggérés (proches du centre) :")
        # Suggérer les produits les plus proches du centre
        generic_products = suggest_products_near_center(model, center, topn)
        for product, distance in generic_products:
            print(f"{product}: {distance:.4f} (distance au centre)")


def main():
    file_path = 'data/ecommerce_store.csv'  # Chemin vers le fichier CSV

    # Charger et nettoyer les données du fichier CSV
    df = load_data(file_path)

    # Convertir les lignes du CSV en une liste de commandes
    orders = df.tolist()

    # Entraîner le modèle Word2Vec
    model = train_word2vec_model(orders)

    # Calculer le centre des vecteurs
    center = calculate_vector_center(model)

    # Exemple de produit à vérifier
    product_to_check = "Wireless Noise-Canceling Headphones"

    # Récupérer les produits similaires ou suggérer des produits génériques
    get_similar_products(model, product_to_check, center)


if __name__ == "__main__":
    main()
