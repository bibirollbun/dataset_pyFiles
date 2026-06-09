from torchvision.models import resnet18, ResNet18_Weights
import torchvision
from torchvision.transforms import v2
import torch
import torch.nn as nn

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)  


train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomCrop(32, padding=4),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

dev_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((32, 32)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_datasets = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
dev_datasets = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=dev_transform)
test_datasets = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=256, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_datasets, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=256, shuffle=True)


classes = train_datasets.classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)  
model = model.to(device)

#Optimizer
lr = 0.1
epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay= 0.0001)


best_accuracy = 0.0
patience = 10
epochs_no_improve = 0
best_model_path = "best_model.pt"

#Training
n_total_steps = len(train_loader)
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate on dev set
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for images, labels in dev_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()
            n_samples += labels.size(0)

    dev_acc = 100.0 * n_correct / n_samples
    print(f"Accuracy on Dev set: {dev_acc:.2f}%")

    # Early stopping check
    if dev_acc > best_accuracy:
        best_accuracy = dev_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience:
        print("Early stopping")
        break

print("Training done")



#Testing

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, predicted = torch.max(output, 1)

        n_samples += labels.size(0)
        n_correct += (predicted==labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.2f}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f}%')

