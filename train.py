"""
Train model.
"""

import os

import hydra
import mlflow
import pandas as pd
import torch
from classification_network import ClassificationNetwork
from encoder import Encoder
from my_dataset import MyDataset
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Main function.
    """

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_data = pd.read_csv("data/train-drug200.csv")

    x_train, y_train = train_data.drop(columns=["Drug"]), train_data["Drug"]
    x_shape = x_train.shape[1]
    num_classes = y_train.nunique()

    column_list = ["Sex", "BP", "Cholesterol"]
    encoder = Encoder(column_list)

    train_data = MyDataset(x_train, y_train, encoder, mode="train")

    train_subset, val_subset = random_split(
        train_data,
        [
            int(len(train_data) * cfg.train.train_size),
            int(len(train_data) * cfg.train.val_size),
        ],
    )

    train_loader = DataLoader(
        dataset=train_subset, shuffle=True, batch_size=cfg.train.batch_size
    )
    val_loader = DataLoader(
        dataset=val_subset, shuffle=False, batch_size=cfg.train.batch_size
    )

    model = ClassificationNetwork(x_shape, num_classes).to(DEVICE)

    min_val_loss = 10000

    mlflow.set_tracking_uri(cfg.mlflow.get("tracking_uri", "http://127.0.0.1:5000"))
    os.environ["MLFLOW_ARTIFACT_ROOT"] = cfg.mlflow.artifact_root
    exp_id = mlflow.set_experiment(f"training-{cfg.model.model_name}").experiment_id
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{1}"):
        mlflow.log_params(cfg)
    for train_data, train_label in train_loader:
        break

    mlflow.models.infer_signature(train_data.numpy(), train_label.numpy())

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(cfg.train.number_epoch):
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
            torch.save(model.state_dict(), "data/model.pt")
            min_val_loss = val_loss

        train_loss = sum_train_loss / len(train_loader.sampler)
        val_loss = sum_val_loss / len(val_loader.sampler)

        train_accuracy = sum_correct_train_prediction / len(train_loader.sampler)
        val_accuracy = sum_correct_val_prediction / len(val_loader.sampler)

        mlflow.log_metric("train_loss", train_loss, epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, epoch)
        mlflow.log_metric("val_loss", val_loss, epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, epoch)

    if cfg.onnx.save:
        with torch.no_grad():
            torch.onnx.export(
                model,
                train_data.to(DEVICE),
                cfg.onnx.path_to_save,
                export_params=True,
                opset_version=15,
                input_names=["INPUTS"],
                output_names=["OUTPUTS"],
                dynamic_axes={
                    "INPUTS": {0: "BATCH_SIZE"},
                    "OUTPUTS": {0: "BATCH_SIZE"},
                },
            )


if __name__ == "__main__":
    main()
