import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from utilities import Utilities
import torch.nn as nn
from transformer import TransformerEncoder,TransformerDecoder
import torch.optim as optim
import torch.nn.functional as F 
import math


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 150  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        return self.net(x)
    
def compute_classifier_accuracy(encoder, classifier, data_loader):
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            encoded, _ = encoder(X)
            outputs = classifier(encoded.mean(dim=1))
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    accuracy = (100 * total_correct / total_samples)
    encoder.train()
    classifier.train()
    return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader. """
    decoderLMmodel.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            if i >= eval_iters:
                break
            
            X, Y = X.to(decoderLMmodel.lm_head.weight.device), Y.to(decoderLMmodel.lm_head.weight.device)
            
            logits, _ = decoderLMmodel(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            
            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    decoderLMmodel.train()
    return perplexity


def main():
    print("Current working directory:", os.getcwd())
    print("Loading data and creating tokenizer ...")
    texts = load_texts('PA2_code/speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "PA2_Code/speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "PA2_Code/speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
    
    encoder = TransformerEncoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    classifier = Classifier(n_embd, n_hidden, n_output).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    
    for epoch in range(epochs_CLS):
        encoder.train()
        classifier.train()
        
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            encoded, attention_maps = encoder(xb)
            logits = classifier(encoded.mean(dim=1))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # CLS training code here
        train_accuracy = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch No: {epoch+1}, Loss: {total_loss/len(train_CLS_loader):.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    utils = Utilities(tokenizer, encoder)
    utils.sanity_check("Sanity Check", block_size)
    # Print number of parameters in encoder
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Number of parameters in encoder: {encoder_params}")
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    
    inputfile = "PA2_code/speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
        
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    #test_LM_dataset_Obama = LanguageModelingDataset(tokenizer, "PA2_code/speechesdataset/test_LM_hbush.txt")

    decoder = TransformerDecoder(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    def load_test_set(file_name, tokenizer, block_size):
        with open(file_name, 'r', encoding='utf-8') as f:
            test_text = f.read()
            test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return test_loader

# In your main function or where you set up your datasets:
    test_sets = {
    'Obama': load_test_set('PA2_Code/speechesdataset/test_LM_obama.txt', tokenizer, block_size),
    'W. Bush': load_test_set('PA2_Code/speechesdataset/test_LM_wbush.txt', tokenizer, block_size),
    'H. Bush': load_test_set('PA2_Code/speechesdataset/test_LM_hbush.txt', tokenizer, block_size)
    }


    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        logits, _ = decoder(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            print(f"Iteration {i+1}, Loss: {loss.item():.4f}")
            for name, test_set in test_sets.items():
                perplexity = compute_perplexity(decoder, test_set,eval_iters)
                print(f"{name} Perplexity: {perplexity:.2f}")
                
    perplexity_training =  compute_perplexity(decoder, train_LM_loader,eval_iters)
    print("Training Perplexity: " ,perplexity_training)   
     
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Number of parameters in decoder: {decoder_params}")

   
    utils = Utilities(tokenizer, decoder)
    utils.sanity_check("This is a test sentence", block_size)
    
if __name__ == "__main__":
    main()
