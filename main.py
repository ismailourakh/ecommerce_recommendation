import pandas as pd
from gensim.models import Word2Vec

# Fonction pour charger les données du fichier CSV
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    return df

# Fonction pour entraîner le modèle Word2Vec
def train_word2vec_model(orders):
    # Entraîner le modèle Word2Vec sur les commandes d'achats
    model = Word2Vec(sentences=orders, vector_size=100, window=5, min_count=1, workers=4)
    model.save("product_word2vec.model")  # Sauvegarder le modèle
    return model

# Fonction pour obtenir les produits similaires à partir du modèle Word2Vec
def get_similar_products(model, product_name):
    try:
        # Trouver les mots les plus similaires au produit donné
        similar_products = model.wv.most_similar(product_name, topn=5)
        print(f"Produits similaires pour '{product_name}':")
        for product, similarity in similar_products:
            print(f"{product}: {similarity}")
    except KeyError:
        print(f"Le produit '{product_name}' n'est pas dans le modèle.")

def main():
    file_path = 'data/ecommerce_store.csv'

    df = load_data(file_path)

    orders = df.values.tolist()

    model = train_word2vec_model(orders)

    product_to_check = "Smartwatch"

    get_similar_products(model, product_to_check)

if __name__ == "__main__":
    main()
