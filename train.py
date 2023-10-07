"""
Train model.
"""

import pandas as pd
import torch
from classification_network import ClassificationNetwork
from encoder import Encoder
from my_dataset import MyDataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, random_split


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    data = pd.read_csv("drug200.csv")

    x_data, y_data = data.drop(columns=["Drug"]), data["Drug"]
    x_shape = x_data.shape[1]
    num_classes = y_data.nunique()

    x_train, _, y_train, _ = train_test_split(
        x_data, y_data, test_size=0.2, stratify=y_data
    )

    column_list = ["Sex", "BP", "Cholesterol"]
    encoder = Encoder(column_list)

    train_data = MyDataset(x_train, y_train, encoder, mode="train")

    train_subset, val_subset = random_split(
        train_data, [int(len(train_data) * 0.8), int(len(train_data) * 0.2)]
    )

    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=16)
    val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=16)

    model = ClassificationNetwork(x_shape, num_classes).to(DEVICE)

    number_epoch = 100
    learning_rate = 0.01
    min_val_loss = 10000

    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    for _ in range(number_epoch):
        sum_train_loss = 0
        sum_val_loss = 0
        sum_correct_train_prediction = 0
        sum_correct_val_prediction = 0

        model.train()
        for train_data, train_label in train_loader:
            train_preds = model(train_data.to(DEVICE))
            train_loss = loss_func(train_preds, train_label.to(DEVICE))

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            sum_train_loss += train_loss.item() * train_data.size(0)

            _, train_prediction = torch.max(train_preds, 1)
            sum_correct_train_prediction += torch.sum(
                train_prediction == train_label.to(DEVICE)
            ).item()

        model.eval()
        for val_data, val_label in val_loader:
            val_preds = model(val_data.to(DEVICE))
            val_loss = loss_func(val_preds, val_label.to(DEVICE))
            sum_val_loss += val_loss.item() * val_data.size(0)

            _, val_prediction = torch.max(val_preds, 1)
            sum_correct_val_prediction += torch.sum(
                val_prediction == val_label.to(DEVICE)
            ).item()
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), "model.pt")
            min_val_loss = val_loss

        mean_train_loss = sum_train_loss / len(train_loader.sampler)
        mean_val_loss = sum_val_loss / len(val_loader.sampler)

        train_loss_list.append(mean_train_loss)
        val_loss_list.append(mean_val_loss)
        train_accuracy_list.append(
            sum_correct_train_prediction / len(train_loader.sampler)
        )
        val_accuracy_list.append(sum_correct_val_prediction / len(val_loader.sampler))


if __name__ == "__main__":
    main()
