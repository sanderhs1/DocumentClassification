#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from functools import partial
import gensim

# Values
dropout = 0.1
hidden_size = 128
max_length = 64
batch_size = 128

# Configure logging
def setup_logger():
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

# Define classifier class
def max_pooling(x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    # Mask the padded values
    x = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
    # Take the maximum along the sequence length dimension (dim=1)
    return torch.max(x, dim=1)[0]

class Model(nn.Module):
    def __init__(self, args, word2vec, n_labels):
        super().__init__()
        self.pad_index = word2vec.get_index("[PAD]")

        # since we don't want to train or store the embedding, we'll use a buffer
        self.embedding = torch.nn.Parameter(torch.FloatTensor(word2vec.vectors), requires_grad=True)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(word2vec.vector_size if i == 0 else hidden_size, hidden_size),
                nn.GELU()
            )
            for i in range(args.num_layers)
        ])
        self.pooling = max_pooling
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_labels)
        )

    def forward(self, input_ids):
        mask = (input_ids == self.pad_index)  # shape: [B, T]

        hidden_vectors = F.embedding(input_ids, weight=self.embedding)  # shape: [B, T, D]
        for hidden_layer in self.hidden_layers:
            hidden_vectors = hidden_layer(hidden_vectors)  # shape: [B, T, D]
        pooled_representation = self.pooling(hidden_vectors, mask)  # shape: [B, D]
        logits = self.output(pooled_representation)  # shape: [B, 2]
        return logits

# Dataset class
class TextDataset(Dataset):
    def __init__(self, data, word2vec, label_vocab=None):
        self.unk_index = word2vec.get_index("[UNK]")  
        self.tokens = [
            [
                word2vec.get_index(token, default=self.unk_index)  
                for token in document.split()  
            ]
            for document in data['lemmatized']  
        ]

        unk_tokens = sum(token == self.unk_index for document in self.tokens for token in document)
        n_tokens = sum(1 for document in self.tokens for token in document)
        print(f"Percentage of unknown tokens: {unk_tokens / n_tokens:.2%}")

        self.label_vocab = list(sorted(data['source'].unique()))  # List of unique sources
        self.label_indexer = {source: idx for idx, source in enumerate(self.label_vocab)}

        self.label = data['source'].map(self.label_indexer)
        self.num_labels = len(self.label_vocab)

    def __getitem__(self, index):
        current_tokens = self.tokens[index]
        current_label = self.label.iloc[index]
        x = torch.LongTensor(current_tokens)
        y = torch.LongTensor([current_label])  
        return x, y

    def __len__(self):
        return len(self.tokens)  

# Load Embeddings
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

# Collate function for batching
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

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation parameters')
    parser.add_argument('--test', type=str, 
                        default='/fp/projects01/ec403/IN5550/obligatories/1/signal20_obligatory1_train.tsv.gz',
                        help='Path to the test set file')
    parser.add_argument('--model', type=str,
                        default='Model/checkpoint.bin',
                        help='Path to the saved model checkpoint')
    parser.add_argument('--embeddings', type=str,
                        default='/fp/projects01/ec403/models/static/40/model.bin',
                        help='Path to pre-trained embeddings')
    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    # Set up logging
    logger = setup_logger()

    # Load embeddings
    word2vec = load_embedding(args.embeddings)
    word2vec["[UNK]"] = torch.tensor(word2vec.vectors).mean(dim=0).numpy()
    word2vec["[PAD]"] = torch.zeros(word2vec.vector_size).numpy()

    # Load the test set
    test_df = pd.read_csv(args.test, sep='\t', header=0, compression='gzip')
    test_df['lemmatized'] = test_df['lemmatized'].apply(preprocess_text)

    # Load the model checkpoint
    checkpoint = torch.load(args.model)
    label_vocab = checkpoint["label_vocabulary"]

    # Prepare the test dataset and dataloader
    test_dataset = TextDataset(test_df, word2vec, label_vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_function, word2vec.get_index("[PAD]"), max_length),
    )

    # Initialize the model
    model = Model(args, word2vec, n_labels=len(label_vocab))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            y_true_all.extend(y_batch.cpu().numpy())
            y_pred_all.extend(predictions.argmax(dim=1).cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1: {f1:.4f}")

if __name__ == "__main__":
    main()
