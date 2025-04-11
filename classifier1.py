# Import necessary libraries for processing and model training
import argparse
import logging
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
import re


# Function to preprocess text data before feeding into the model
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase to ensure uniformity
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation marks
    text = re.sub(r'\d+', '', text)  # Remove numerical values from the text
    text = ' '.join([word for word in text.split() if len(word) > 2])  # Remove short words (length <= 2) as they may be uninformative
    text = re.sub(r'\s+', ' ', text).strip()  # Remove unnecessary spaces and trim the text
    return text  # Return the cleaned text

# Define a Multi-Layer Perceptron (MLP) classifier for text classification
class MlpClassifier(nn.Module):
    def __init__(self, feature_size, hidden_size=128, num_classes=None):  
        super().__init__()
        self.hidden1 = nn.Linear(feature_size, hidden_size)  # First hidden layer that connects input to hidden neurons
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization to stabilize and speed up training
        self.activation = nn.LeakyReLU()  # Leaky ReLU activation function to introduce non-linearity
        self.dropout = nn.Dropout(p=0.5)  # Dropout to reduce overfitting by randomly disabling neurons during training
        self.output = nn.Linear(hidden_size, num_classes)  # Final output layer for classification
    
    def forward(self, x):
        x = self.hidden1(x)  # Apply first hidden layer transformation
        x = self.bn1(x)  # Normalize hidden layer output
        x = self.activation(x)  # Apply activation function
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.output(x)  # Get final classification output
        return x  # Return the final output logits

# Dataset class to handle feature-label pairs in PyTorch training
class TextDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)  # Convert features into PyTorch tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels into PyTorch tensor
    
    def __len__(self):
        return len(self.labels)  # Return total number of samples in the dataset
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  # Return a single sample (feature, label) pair

# Main execution block (runs only when script is executed directly)
if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)  # Set up logging for monitoring execution

    # Argument parsing to allow configuration through command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the dataset", required=True)
    parser.add_argument("--vocab_size", help="Maximum size of the vocabulary", type=int, default=10000)
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=15)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.001)
    args = parser.parse_args()

    # Load dataset
    data_df = pd.read_csv(args.dataset, sep='\t')  
    train_data, valid_data = train_test_split(data_df, test_size=0.1, random_state=42, stratify=data_df["source"])
    num_classes = len(train_data["source"].unique())  # Determine number of unique class labels
    logger.info(f"Train size: {len(train_data)}, Validation size: {len(valid_data)}")

    # Preprocess the text data for training and validation
    train_data['lemmatized'] = train_data['lemmatized'].apply(preprocess_text)
    valid_data['lemmatized'] = valid_data['lemmatized'].apply(preprocess_text)

    # Convert text data into numerical representations using CountVectorizer
    #vectorizer = TfidfVectorizer(max_features=args.vocab_size, ngram_range=(1, 2))
    vectorizer = CountVectorizer(binary = True, max_features=args.vocab_size, ngram_range=(1, 2))
    #vectorizer = CountVectorizer(max_features=args.vocab_size, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_data['lemmatized']).toarray()
    X_valid = vectorizer.transform(valid_data['lemmatized']).toarray()

    # Encode labels to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['source'])
    y_valid = label_encoder.transform(valid_data['source'])

    # Create PyTorch dataset objects and dataloaders for training and validation
    train_dataset = TextDataset(X_train, y_train)
    valid_dataset = TextDataset(X_valid, y_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)

    # Initialize model, loss function, optimizer, and learning rate scheduler
    model = MlpClassifier(feature_size=X_train.shape[1], hidden_size=128, num_classes=num_classes)
    class_counts = train_data['source'].value_counts()
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()
    weights = torch.tensor(class_weights.values, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)  # Weighted cross-entropy loss to handle imbalanced classes
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)  # Adam optimizer with weight decay to prevent overfitting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # Reduce learning rate if validation loss plateaus

    # Training loop with early stopping
    logger.info("Starting training...")

    best_val_loss = float('inf')
    patience = 3 
    trigger_times = 0

    y_true_all = []
    y_pred_all = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            y_true_all.extend(y_batch.numpy())
            y_pred_all.extend(predictions.argmax(dim=1).numpy())

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation-phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(valid_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)

    logger.info(f"Final Metrics - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")



