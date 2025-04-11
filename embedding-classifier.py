import argparse  
import logging 
import os  
import pandas as pd 
import re  
import torch  
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader  
from torch.optim import AdamW  
from functools import partial  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
import gensim 
from typing import List, Tuple  

# Hyperparameters and configurations
dropout = 0.1  # Dropout rate for regularization
hidden_size = 128  # Hidden layer size in neural network
max_length = 64  # Maximum length of tokenized input sequences
batch_size = 128  # Batch size for training
lr = 0.01  # Learning rate for optimization

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove short words (less than 3 characters)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Logger setup function
def setup_logger():
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


# Pooling functions for sentence representations
def mean_pooling(x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    # Mask padding tokens by setting them to zero
    x = x.masked_fill(mask.unsqueeze(-1), 0.0)
    # Compute mean pooling by summing embeddings and dividing by the valid token count
    return torch.sum(x, dim=1) / (mask == False).sum(dim=1).unsqueeze(-1)

def sum_pooling(x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    # Mask padding tokens by setting them to zero
    x = x.masked_fill(mask.unsqueeze(-1), 0.0)
    # Compute sum pooling by summing embeddings across the sequence
    return torch.sum(x, dim=1)

def max_pooling(x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    # Mask padding tokens by setting them to negative infinity
    x = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
    # Compute max pooling by taking the max value across the sequence
    return torch.max(x, dim=1)[0]


# Neural network model definition
class Model(nn.Module):
    def __init__(self, args, word2vec, n_labels):
        super().__init__()
        self.pad_index = word2vec.get_index("[PAD]")  # Index for padding token

        # Initialize word embedding layer using pre-trained embeddings
        # since we don't want to train or store the embedding, we'll use a buffer
        #self.embedding = torch.nn.Parameter(torch.FloatTensor(word2vec.vectors), requires_grad=True)
        self.register_buffer('embedding', torch.FloatTensor(word2vec.vectors), persistent=False)


        # Create multiple hidden layers using dropout, linear transformations, and GELU activation
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(word2vec.vector_size if i == 0 else hidden_size, hidden_size),
                nn.GELU()
            )
            for i in range(args.num_layers)
        ])
        # Set pooling method to max pooling
        self.pooling = max_pooling
        # Output layer for classification
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_labels)
        )

    def forward(self, input_ids):
        mask = (input_ids == self.pad_index)  # shape: [B, T] - create mask for padding tokens
        hidden_vectors = F.embedding(input_ids, weight=self.embedding)  # shape: [B, T, D]  - convert token indices to embeddings
        for hidden_layer in self.hidden_layers:
            hidden_vectors = hidden_layer(hidden_vectors)  # shape: [B, T, D]  - pass through hidden layers
        pooled_representation = self.pooling(hidden_vectors, mask)  # shape: [B, D] - apply pooling
        logits = self.output(pooled_representation)  # shape: [B, 2] - generate output logits
        return logits

# Dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, data, word2vec, label_vocab=None):
        self.unk_index = word2vec.get_index("[UNK]")  # Index for unknown token

        # Tokenize the text 
        self.tokens = [
            [
                word2vec.get_index(token, default=self.unk_index)  # Look up the full token
                for token in document.split()  # Split by spaces into tokens like "market_VERB"
            ]
            for document in data['lemmatized']  # Process the text column
        ]

        # Calculate the percentage of unknown tokens
        unk_tokens = sum(token == self.unk_index for document in self.tokens for token in document)
        n_tokens = sum(1 for document in self.tokens for token in document)
        print(f"Percentage of unknown tokens: {unk_tokens / n_tokens:.2%}")

        # Create a vocabulary of unique sources (labels)
        self.label_vocab = list(sorted(data['source'].unique()))  # List of unique sources
        self.label_indexer = {source: idx for idx, source in enumerate(self.label_vocab)}

        # Map the sources to their corresponding indices
        self.label = data['source'].map(self.label_indexer)
        self.num_labels = len(self.label_vocab)

    def __getitem__(self, index):
        # Get the tokens and label for a specific index
        current_tokens = self.tokens[index]
        current_label = self.label.iloc[index]

        # Convert tokens to a tensor
        x = torch.LongTensor(current_tokens)
        y = torch.LongTensor([current_label])  # Convert the label to a tensor
        return x, y

    def __len__(self):
        return len(self.tokens)  # Return the length of the dataset (number of documents)  

# Function to load word embeddings
def load_embedding(modelfile):
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    elif modelfile.endswith(".txt.gz") or modelfile.endswith(".txt") or modelfile.endswith(".vec.gz") or modelfile.endswith(".vec"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    elif modelfile.endswith("parameters.bin"):
        emb_model = gensim.models.fasttext.load_facebook_vectors(modelfile)
    else:
        emb_model = gensim.models.KeyedVectors.load(modelfile)

    return emb_model

# Collate function to pad and batch input sequences
def collate_function(padding_index: int, max_length: int, samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = [x for x, _ in samples]
    labels = [y for _, y in samples]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True,
        padding_value=padding_index, 
    )
    input_ids_padded = input_ids_padded[:, :max_length]
    return input_ids_padded, torch.LongTensor(labels)

# Argument parser for command-line execution
def parse_args():
    parser = argparse.ArgumentParser(description='Model training parameters')
    parser.add_argument('--path', type=str, 
                        default='/fp/projects01/ec403/IN5550/obligatories/1/signal20_obligatory1_train.tsv.gz',
                        help='Path to the dataset')
    parser.add_argument('--embeddings', type=str,
                        default='/fp/projects01/ec403/models/static/40/model.bin',
                        help='Path to pre-trained embeddings')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    return parser.parse_args()

# Main function
def main():
    # Parse command-line arguments
    args = parse_args()

    # Set up logging to track execution details
    logger = setup_logger()

    # Load pre-trained word embeddings
    word2vec = load_embedding(args.embeddings)
    word2vec["[UNK]"] = torch.tensor(word2vec.vectors).mean(dim=0).numpy()
    word2vec["[PAD]"] = torch.zeros(word2vec.vector_size).numpy()

    # Use a random vector for unknown tokens and a zero vector for padding tokens
    #word2vec["[UNK]"] = torch.randn(word2vec.vector_size).numpy()
    #word2vec["[PAD]"] = torch.zeros(word2vec.vector_size).numpy()

    # Load the dataset from the specified path
    df = pd.read_csv(args.path, sep='\t', header=0, compression='gzip')
    #comment out below for text_NOUN
    df['lemmatized'] = df['lemmatized'].apply(preprocess_text)

    # Split the dataset into training and validation sets (90% train, 10% validation)
    train_df, val_df = train_test_split(df, train_size=0.9, random_state=42, stratify=df["source"])
     # Create training and validation datasets
    train_dataset = TextDataset(train_df, word2vec)
    val_dataset = TextDataset(val_df, word2vec, label_vocab=train_dataset.label_vocab)
    # Create data loaders for training and validation sets
    train_iter = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2,
        collate_fn=partial(collate_function, word2vec.get_index("[PAD]"), max_length)
    )
    val_iter = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=2,
        collate_fn=partial(collate_function, word2vec.get_index("[PAD]"), max_length)
    )

    # Initialize the model with the specified arguments, embeddings, and number of labels
    model = Model(
        args,
        word2vec=word2vec,
        n_labels=len(train_dataset.label_vocab)
    )
    # Initialize the AdamW optimizer with the model's trainable parameters
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0
    )
    # Initialize a cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.epochs * len(train_iter)
    )
    
    # Training loop over the specified number of epochs
    for epoch in range(args.epochs):
        # Set the model to training mode
        model.train()
        total_loss = 0
        y_true_all, y_pred_all = [], []
        # Iterate over batches in the training data loader
        for X_batch, y_batch in train_iter:
            optimizer.zero_grad() # Reset gradients 
            predictions = model(X_batch) # Forward pass
            loss = F.cross_entropy(predictions, y_batch) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update model parameter
            total_loss += loss.item() # Accumulate loss

            # Accumulate true labels and predicted labels for evaluation
            y_true_all.extend(y_batch.cpu().numpy())
            y_pred_all.extend(predictions.argmax(dim=1).cpu().numpy())
            
        # Compute average training loss and evaluation metrics
        avg_train_loss = total_loss / len(train_iter)
        train_accuracy = accuracy_score(y_true_all, y_pred_all)
        train_precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        train_recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        train_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        # Log training metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train loss: {avg_train_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, "
                    f"Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")

        # Validation loop
        model.eval() # Set the mdodel to evaluation mode
        y_true_all, y_pred_all = [], []
        with torch.no_grad(): # Disable gradient computation for validation
            for X_batch, y_batch in val_iter:
                predictions = model(X_batch)
                y_true_all.extend(y_batch.cpu().numpy()) # Accumulate true labels 
                y_pred_all.extend(predictions.argmax(dim=1).cpu().numpy()) # Accumulate predicted labels

        # Compute validation metrics
        val_accuracy = accuracy_score(y_true_all, y_pred_all) 
        val_precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        val_recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        val_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        # Log validation metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Validation Accuracy: {val_accuracy:.4f}, "
                    f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, "
                    f"Validation F1: {val_f1:.4f}")



    



if __name__ == "__main__":
    main()
