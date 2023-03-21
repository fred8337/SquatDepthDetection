import torch
import numpy as np
import torchvision.models
from torch.utils.data import DataLoader, Subset
from Data.SquatDataset import SquatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
epochs = 100
lr = 1e-3

# Load dataset
dataset = SquatDataset()

# Split data into training and test sets
indices = np.arange(len(dataset))
np.random.seed(0)
np.random.shuffle(indices)
split = int(0.8 * len(dataset))
train_indices, test_indices = indices[:split], indices[split:]

train_dataloader = DataLoader(Subset(dataset=dataset, indices=train_indices), batch_size=16, shuffle=True)
test_dataloader = DataLoader(Subset(dataset=dataset, indices=test_indices), batch_size=16, shuffle=False)

# Load densenet model

model = torchvision.models.densenet121()
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(1024, 1),
    torch.nn.Sigmoid()
)
model.to(device)

# Load optimizer

optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = None
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=1e-2, cycle_momentum=False, step_size_up=200)

loss_fn = torch.nn.functional.binary_cross_entropy

# Train model

best_loss = np.inf
best_accuracy = 0
best_model = None

for epoch in range(epochs):
    loss_for_epoch = None
    model.train(True)
    for data in train_dataloader:
        # data.to(device)
        inputs, labels = inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if loss_for_epoch is not None:
            loss_for_epoch += loss.item()
        else:
            loss_for_epoch = loss.item()
    val_loss_for_epoch = None
    val_accuracy = None
    val_number_of_items = 0
    model.train(False)
    for data in test_dataloader:
        # data.to(device)
        inputs, labels = inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        if val_loss_for_epoch is not None:
            val_loss_for_epoch += loss.item()
        else:
            val_loss_for_epoch = loss.item()
        if val_accuracy is not None:
            predictions = outputs > 0.5
            val_accuracy += torch.sum(predictions==labels)
        else:
            predictions = outputs > 0.5
            val_accuracy = torch.sum(predictions==labels)
        val_number_of_items += len(labels)
    val_accuracy = val_accuracy/val_number_of_items
    # if best_loss > val_loss_for_epoch:
    #     print(
    #         f"Loss improved at epoch {epoch} from {best_loss} to {val_loss_for_epoch}, with in sample loss {loss_for_epoch}")
    #     best_loss = val_loss_for_epoch
    #     best_model = model.state_dict()
    #     torch.save(best_model, "best_model.pt")
    print(f"Accuracy: {val_accuracy.item()}, loss: {loss_for_epoch}, val_loss: {val_loss_for_epoch}")
    if best_accuracy < val_accuracy.item():
        print(
            f"Accuracy improved at epoch {epoch} from {best_accuracy} to {val_accuracy}, with val loss {val_loss_for_epoch}")
        best_accuracy = val_accuracy.item()
        best_model = model.state_dict()
        torch.save(best_model, "best_model2.pt")

from TestOnVideo import FrameEvaluator


