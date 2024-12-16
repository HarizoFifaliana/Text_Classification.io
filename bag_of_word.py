import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_files


class BagOfWordsLogisticRegression:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words="english")
        self.model = LogisticRegression(max_iter=1000)

    def preprocess(self, texts):
        """
        Transforme les textes en une représentation bag-of-words.
        """
        return self.vectorizer.fit_transform(texts)

    def train(self, texts, labels):
        """
        Entraîne la régression logistique avec des données de texte et leurs labels.
        """
        X = self.preprocess(texts)
        self.model.fit(X, labels)
        return X

    def predict(self, texts):
        """
        Prédit les labels pour de nouveaux textes.
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def evaluate(self, texts, labels):
        """
        Évalue le modèle sur des données de test.
        """
        predictions = self.predict(texts)
        acc = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        return acc, report


def main():
    # Charger le dataset IMDb (pré-chargé dans scikit-learn)
    print("Chargement des données IMDb...")
    dataset = load_files('aclImdb', categories=['pos', 'neg'], shuffle=True, random_state=42)
    texts, labels = dataset.data, dataset.target

    # Convertir les textes en chaînes de caractères (nécessaire pour le dataset IMDb)
    texts = [text.decode('utf-8') for text in texts]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Initialiser la classe et entraîner le modèle
    print("Entraînement du modèle...")
    classifier = BagOfWordsLogisticRegression()
    classifier.train(X_train, y_train)

    # Évaluation du modèle
    print("Évaluation sur l'ensemble de test...")
    accuracy, report = classifier.evaluate(X_test, y_test)

    print(f"Précision : {accuracy:.2f}")
    print("Rapport de classification :")
    print(report)


if __name__ == "__main__":
    main()
