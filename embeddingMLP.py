import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm


# Pré-traitement des données et dataset personnalisé
class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(text)
        return torch.tensor(tokenized[:self.max_len], dtype=torch.long), torch.tensor(label, dtype=torch.float)


# Tokenizer pour créer un vocabulaire
class Tokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = Counter()

    def fit_on_texts(self, texts):
        for text in texts:
            self.vocab.update(text.split())
        most_common = self.vocab.most_common(self.vocab_size - 1)  # Réserver un index pour <UNK>
        self.word2idx = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
        self.word2idx["<UNK>"] = 0
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def texts_to_sequences(self, texts):
        return [[self.word2idx.get(word, 0) for word in text.split()] for text in texts]

    def __call__(self, text):
        return self.texts_to_sequences([text])[0]


# Modèle PyTorch
class MLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_len):
        super(MLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * max_len, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Entraînement"):
        texts, labels = batch
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)


def main():
    # Charger et pré-traiter les données
    print("Chargement des données IMDb...")
    dataset = load_files('aclImdb', categories=['pos', 'neg'], shuffle=True, random_state=42)
    texts, labels = dataset.data, dataset.target
    texts = [text.decode('utf-8') for text in texts]

    # Préparer le tokenizer
    tokenizer = Tokenizer(vocab_size=10000)
    tokenizer.fit_on_texts(texts)

    # Tokenization
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = 200
    labels = torch.tensor(labels, dtype=torch.float)

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # Convertir en datasets et dataloaders
    train_dataset = IMDbDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = IMDbDataset(X_test, y_test, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=pad_sequence)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=pad_sequence)

    # Paramètres du modèle
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPModel(vocab_size, embedding_dim, hidden_dim, max_len).to(device)

    # Optimiseur et fonction de perte
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Entraîner le modèle
    print("Entraînement du modèle...")
    for epoch in range(5):
        train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
        print(f"Époque {epoch + 1}, Perte d'entraînement : {train_loss:.4f}")

    # Évaluer le modèle
    print("Évaluation sur l'ensemble de test...")
    test_loss, accuracy = evaluate_model(model, test_dataloader, criterion, device)
    print(f"Perte de test : {test_loss:.4f}, Précision : {accuracy:.2f}")


if __name__ == "__main__":
    main()
