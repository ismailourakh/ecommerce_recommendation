from flask import Flask, request, jsonify
from gensim.models import Word2Vec
import numpy as np

app = Flask(__name__)

# Load the Word2Vec model
model = Word2Vec.load("product_word2vec.model")

# Calculate the center of the vectors

center = np.mean([model.wv[word] for word in model.wv.index_to_key], axis=0)

def suggest_products_near_center(model, center, topn=5):
    distances = {
        word: np.linalg.norm(model.wv[word] - center)
        for word in model.wv.index_to_key
    }
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    return sorted_distances[:topn]

@app.route('/get_similar_products', methods=['GET'])
def get_similar_products_api():
    product_name = request.args.get('product_name')
    topn = int(request.args.get('topn', 5))

    try:
        similar_products = model.wv.most_similar(product_name, topn=topn)
        return jsonify({
            "product_name": product_name,
            "similar_products": similar_products
        })
    except KeyError:
        generic_products = suggest_products_near_center(model, center, topn)
        return jsonify({
            "product_name": product_name,
            "error": f"'{product_name}' not found in the model.",
            "generic_products": generic_products
        })

if __name__ == '__main__':
    app.run(debug=True)
# http://127.0.0.1:5000/get_similar_products?product_name=Smartwatch&topn=5