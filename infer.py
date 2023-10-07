"""
Inference model.
"""

import pandas as pd
import torch
from classification_network import ClassificationNetwork
from encoder import Encoder, encoder_target
from my_dataset import MyDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    """
    Main function.
    """
    data = pd.read_csv("drug200.csv")

    x_data, y_data = data.drop(columns=["Drug"]), data["Drug"]
    x_shape = x_data.shape[1]
    num_classes = y_data.nunique()

    column_list = ["Sex", "BP", "Cholesterol"]
    encoder = Encoder(column_list)

    _, x_test, _, y_test = train_test_split(
        x_data, y_data, test_size=0.2, stratify=y_data
    )

    test_data = MyDataset(x_test, y_test, encoder, mode="test")
    test_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=16)

    model = ClassificationNetwork(x_shape, num_classes).to(DEVICE)

    model.load_state_dict(torch.load("model.pt"))
    sum_correct_test_prediction = 0

    test_preds_list = []

    model.eval()
    for test_data in test_loader:
        test_preds_list.append(model.forward(test_data.to(DEVICE)))

    test_preds = torch.cat(test_preds_list, dim=0)

    test_label = torch.tensor(encoder_target(y_test))

    _, prediction = torch.max(test_preds, 1)
    sum_correct_test_prediction += torch.sum(prediction == test_label.to(DEVICE)).item()

    preds_data = pd.DataFrame(
        data={
            "index": y_test.index,
            "true_target": y_test,
            "preds": encoder.encoder_target(prediction, inverse=True),
        }
    )
    preds_data.to_csv("result.csv")

    print(f"Accuracy: {sum_correct_test_prediction / len(test_loader.sampler)}")


if __name__ == "__main__":
    main()
