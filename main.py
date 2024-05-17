import torch
import torch.nn as nn
import Utils

from models.LLAModel import LLAModel
    
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    
    
    
if __name__ == '__main__':
    
    # Example usage
    vocab_size = 10000  # Example vocabulary size
    embedding_dim = 300
    hidden_size = 128
    num_topics = 50
    batch_size = 64
    lr = 0.001
    epochs = 5

    # Create the LLA model
    lla_model = LLAModel(vocab_size, embedding_dim, hidden_size, num_topics)

    # Example training data (you would replace this with your actual data)
    train_data = [torch.randint(0, vocab_size, (batch_size,), dtype=torch.long) for _ in range(10)]

    # Train the model
    lla_model.train_model(train_data, epochs, batch_size, lr)

    # Run the EM algorithm
    lla_model.em_algorithm(train_data, num_iterations=10, batch_size=batch_size, lr=lr)