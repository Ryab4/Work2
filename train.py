import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Устройство: {device}\n')

# Загрузка датасета
transform_basic = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_basic)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform_basic)

print(f'Обучающая выборка: {len(train_dataset):,}')
print(f'Тестовая выборка: {len(test_dataset):,}')
print(f'Количество классов: {len(train_dataset.classes)}\n')

# Визуализация примеров
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i * 1000]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Класс: {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=150)
plt.close()
print('✓ Примеры датасета сохранены: dataset_samples.png\n')

# Трансформации
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_299 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DataLoaders
train_224 = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_224)
test_224 = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform_224)
train_299 = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_299)
test_299 = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform_299)

train_loader_224 = DataLoader(train_224, batch_size=64, shuffle=True, num_workers=2)
test_loader_224 = DataLoader(test_224, batch_size=64, shuffle=False, num_workers=2)
train_loader_299 = DataLoader(train_299, batch_size=32, shuffle=True, num_workers=2)
test_loader_299 = DataLoader(test_299, batch_size=32, shuffle=False, num_workers=2)

# Функции обучения
def train_epoch(model, loader, criterion, optimizer, device, is_inception=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if is_inception:
            outputs, aux = model(inputs)
            loss = criterion(outputs, labels) + 0.4 * criterion(aux, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

def train_model(model, train_loader, test_loader, name, epochs=10, is_inception=False):
    print(f'{"="*70}\nОбучение: {name}\n{"="*70}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'time': []}
    
    for epoch in range(epochs):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, is_inception)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_time = time.time() - start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['time'].append(epoch_time)
        
        print(f'Эпоха {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | Время: {epoch_time:.1f}с')
    
    print(f'Завершено! Лучшая точность: {max(history["test_acc"]):.2f}%\n')
    return history

# Обучение моделей
NUM_EPOCHS = 10
NUM_CLASSES = 47
results = {}

# ResNet-18
print('\n[1/4] ResNet-18')
resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
results['ResNet-18'] = train_model(resnet, train_loader_224, test_loader_224, 'ResNet-18', NUM_EPOCHS)
del resnet
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# VGG-16
print('\n[2/4] VGG-16')
vgg = models.vgg16(pretrained=False)
vgg.classifier[6] = nn.Linear(4096, NUM_CLASSES)
results['VGG-16'] = train_model(vgg, train_loader_224, test_loader_224, 'VGG-16', NUM_EPOCHS)
del vgg
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Inception v3
print('\n[3/4] Inception v3')
inception = models.inception_v3(pretrained=False, aux_logits=True)
inception.fc = nn.Linear(inception.fc.in_features, NUM_CLASSES)
inception.AuxLogits.fc = nn.Linear(inception.AuxLogits.fc.in_features, NUM_CLASSES)
results['Inception-v3'] = train_model(inception, train_loader_299, test_loader_299, 'Inception-v3', NUM_EPOCHS, True)
del inception
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# DenseNet-161
print('\n[4/4] DenseNet-161')
densenet = models.densenet161(pretrained=False)
densenet.classifier = nn.Linear(densenet.classifier.in_features, NUM_CLASSES)
results['DenseNet-161'] = train_model(densenet, train_loader_224, test_loader_224, 'DenseNet-161', NUM_EPOCHS)
del densenet
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Сравнение обучения моделей на EMNIST Balanced', fontsize=16)

# Train Loss
ax = axes[0, 0]
for name, hist in results.items():
    ax.plot(range(1, len(hist['train_loss'])+1), hist['train_loss'], marker='o', label=name, linewidth=2)
ax.set_xlabel('Эпоха')
ax.set_ylabel('Loss')
ax.set_title('Train Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Test Loss
ax = axes[0, 1]
for name, hist in results.items():
    ax.plot(range(1, len(hist['test_loss'])+1), hist['test_loss'], marker='o', label=name, linewidth=2)
ax.set_xlabel('Эпоха')
ax.set_ylabel('Loss')
ax.set_title('Test Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Train Accuracy
ax = axes[1, 0]
for name, hist in results.items():
    ax.plot(range(1, len(hist['train_acc'])+1), hist['train_acc'], marker='o', label=name, linewidth=2)
ax.set_xlabel('Эпоха')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Train Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Test Accuracy
ax = axes[1, 1]
for name, hist in results.items():
    ax.plot(range(1, len(hist['test_acc'])+1), hist['test_acc'], marker='o', label=name, linewidth=2)
ax.set_xlabel('Эпоха')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Test Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
plt.close()
print('\n✓ Графики сохранены: training_results.png')

# Таблица результатов
print(f'\n{"="*95}')
print(f'{"Модель":<15} | {"Train Loss":>11} | {"Test Loss":>11} | {"Train Acc":>11} | {"Test Acc":>11} | {"Время/эп":>10}')
print("="*95)
for name, hist in results.items():
    print(f'{name:<15} | {hist["train_loss"][-1]:>11.4f} | {hist["test_loss"][-1]:>11.4f} | '
          f'{hist["train_acc"][-1]:>10.2f}% | {hist["test_acc"][-1]:>10.2f}% | {np.mean(hist["time"]):>9.1f}с')
print("="*95)

print('\n✅ Эксперимент завершён!')
