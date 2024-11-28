# E-Commerce Recommendation System

This project is a recommendation system for an e-commerce platform. It uses Word2Vec to find product similarities and suggest relevant products based on customer purchases.

## Features

- Recommends similar products using a trained Word2Vec model.
- Processes purchase data from CSV files.
- Offers insight into product relationships based on purchase history.

## Project Structure

```plaintext
ecommerce_recommendation/
│
├── main.py                # Main script to run the recommendation system
├── data/                  # Folder containing data files
│   └── ecommerce_store.csv  # Example CSV file with purchase data
├── product_word2vec.model # Word2Vec model for product recommendations
├── requirements.txt       # Dependencies for the project
├── word2vec_model.model   # Backup or alternative Word2Vec model
└── .gitignore             # Git configuration to ignore unnecessary files


---

## Requirements

- Python 3.x
- Required libraries: `pandas`, `gensim`, `numpy`

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ismailourakh/ecommerce_recommendation.git

## Navigate to the Project Folder:


cd ecommerce_recommendation
## Set Up a Virtual Environment (Optional):


python -m venv .venv
source .venv/bin/activate       # On macOS/Linux
.venv\Scripts\activate          # On Windows
## Install Dependencies:


pip install -r requirements.txt
## Usage
## Prepare Your Data:

## Ensure you have a CSV file in the data/ folder, formatted like this:

0,1,2,3,4,5,6
USB Drive,,,,,,
Keyboard,Smartwatch,Camera,USB Drive,Laptop,,
Laptop,Tablet,Mouse,Headphones,Camera,Keyboard,Smartphone
## Run the Program:


python main.py
## Sample Output:


Produits similaires pour 'Smartwatch':
Charger: 0.2251
USB Drive: 0.2064
Mouse: 0.1911
