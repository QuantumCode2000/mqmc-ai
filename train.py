import numpy as np
import random
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Recorra cada oración en nuestros patrones de intenciones
for intent in intents['intents']:
    tag = intent['tag']
    # agregar a la lista de etiquetas
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenizar cada palabra en la oración
        w = tokenize(pattern)
        # añade a la lista de las palabras
        all_words.extend(w)
        # agregar al par xy
        xy.append((w, tag))

# raíz y baje cada palabra
ignore_words = ['?', '.', '!', ',','¿']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# eliminar duplicados y ordenar
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# crear datos de entrenamiento
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bolsa de palabras para cada frase_patrón
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch Pérdida de entropía cruzada solo necesita etiquetas de clase, no one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hiperparámetros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # admite indexación de modo que el conjunto de datos [i] se pueda utilizar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # podemos llamar a len(conjunto de datos) para devolver el tamañoe
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Pérdida y optimizador/función de activación.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
accuracies = []
# Entrenar el modelo
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for (words, labels) in train_loader:
        
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Pase adelantado
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Retroceder y optimizar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        #  Realice un seguimiento de las predicciones correctas y del número total de muestras
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_accuracy = correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
        # the accuracy is not correct, because the batch size is 8, so the accuracy is calculated for each batch
        # this result is not the final accuracy
        
        
    # if (epoch+1) % 100 == 0:
        # print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        #  print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')


# print(f'final loss: {loss.item():.4f}')
# print(f'Final loss: {loss.item():.4f}, Final accuracy: {accuracy:.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')


# Plotting loss
plt.plot(losses)
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida de entrenamiento')
plt.show()

# Plotting accuracy
plt.plot(accuracies)
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión del entrenamiento')
plt.show()

# Train the model
# for epoch in range(num_epochs):
#     correct = 0
#     total = 0
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)
        
#         # Forward pass
#         outputs = model(words)
#         # Calculate loss
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Track correct predictions and total number of samples
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#     accuracy = correct / total
    
#     if (epoch+1) % 100 == 0:
#         print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# print(f'Final loss: {loss.item():.4f}, Final accuracy: {accuracy:.4f}')
