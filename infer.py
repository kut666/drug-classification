"""
Inference model.
"""

import hydra
import pandas as pd
import torch
from classification_network import ClassificationNetwork
from encoder import Encoder, encoder_target
from my_dataset import MyDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Main function.
    """

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    test_data = pd.read_csv("data/test-drug200.csv")

    x_test, y_test = test_data.drop(columns=["Drug"]), test_data["Drug"]
    x_shape = x_test.shape[1]
    num_classes = y_test.nunique()

    column_list = ["Sex", "BP", "Cholesterol"]
    encoder = Encoder(column_list)

    test_data = MyDataset(x_test, y_test, encoder, mode="test")
    test_loader = DataLoader(
        dataset=test_data, shuffle=False, batch_size=cfg.infer.batch_size
    )

    model = ClassificationNetwork(x_shape, num_classes).to(DEVICE)

    model.load_state_dict(torch.load("data/model.pt"))

    test_preds_list = []

    model.eval()
    for test_data in test_loader:
        test_preds_list.append(model.forward(test_data.to(DEVICE)))

    test_preds = torch.cat(test_preds_list, dim=0)

    test_label = torch.tensor(encoder_target(y_test))

    _, prediction = torch.max(test_preds, 1)
    sum_correct_test_prediction = torch.sum(
        prediction.to(DEVICE) == test_label.to(DEVICE)
    ).item()

    preds_data = pd.DataFrame(
        data={
            "index": y_test.index,
            "true_target": y_test,
            "preds": encoder_target(prediction.cpu().numpy(), inverse=True),
        }
    )
    preds_data.to_csv("data/result.csv")

    print(f"Accuracy: {sum_correct_test_prediction / len(test_loader.sampler)}")


if __name__ == "__main__":
    main()
