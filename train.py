import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json',encoding="utf8") as f:
    intents = json.load(f)

all_words = [] # all words stemmed form are stored for creating bag_of_words
tags = [] # list of all tags in dataset. 
xy = [] # tuple containing (sentence,tag)

# loop through each sentence in our intents(dataset) patterns(questions)
for intent in intents['intents']:
    tag = intent['tag']
    # adding tag to tags list
    tags.append(tag)
    
    for pattern in intent['patterns']:
        # tokenize each word in the sentence using nltk_utils tokenize method
        w = tokenize(pattern)
        # adding tokenized words to our words list
        # if we do append it would create a 2d list because w is also list
        # so we extend it
        all_words.extend(w)
        # adding tuple of (tokenized_word_array, it's tag) to xy pair
        # example, xy = [(["hi","good","morning"],"greetings")]
        xy.append((w, tag))

# stem and lowering of words in all_words
ignore_words = ['?', '.', '!'] # to ignore from tokenized words
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "Total patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "All unique stemmed words:", all_words)

# create training data
X_train = [] # 2d array
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    # pattern_sentence is tokenized words and all_words are stemmed and tokenized
    # pattern_sentence is stemmed in bag_of_words function
    # returns array of length = size of all_words with 1 and 0's
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) 
    # y: needs only class labels
    label = tags.index(tag) # index of current tag in tags list
    # label is the index of current tag in tags array.
    # like tags = ["greeting","service","goodbye"] and current tag is "goodbye"
    # then label has value 2 and is appended in y_train
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0]) # len is constant as we are using bag_of_words size
hidden_size = 8
output_size = len(tags) # number of output classes
print("input size = {}, hidden size = {}, output size={}".format(input_size, hidden_size, output_size))

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# set device as gpu if available else cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}]')


data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

# saving for later use 
FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved as {FILE}')
