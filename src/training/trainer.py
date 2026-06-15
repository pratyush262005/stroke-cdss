import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(
        self,
        dataloader
    ):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader)

        for batch in pbar:

            image = batch["image"].to(
                self.device
            )

            symmetry = batch[
                "symmetry_map"
            ].to(
                self.device
            )

            labels = batch[
                "label"
            ].to(
                self.device
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                image,
                symmetry
            )

            logits = outputs[
                "logits"
            ]

            loss = self.criterion(
                logits,
                labels
            )

            loss.backward()

            self.optimizer.step()

            running_loss += (
                loss.item()
            )

            preds = logits.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                preds == labels
            ).sum().item()

            pbar.set_description(

                f"Loss: {loss.item():.4f}"

            )

        accuracy = (
            correct / total
        )

        epoch_loss = (
            running_loss
            /
            len(dataloader)
        )

        return {

            "loss": epoch_loss,
            "accuracy": accuracy

        }

    @torch.no_grad()
    def validate(
        self,
        dataloader
    ):

        self.model.eval()

        running_loss = 0.0

        correct = 0
        total = 0

        for batch in dataloader:

            image = batch[
                "image"
            ].to(
                self.device
            )

            symmetry = batch[
                "symmetry_map"
            ].to(
                self.device
            )

            labels = batch[
                "label"
            ].to(
                self.device
            )

            outputs = self.model(
                image,
                symmetry
            )

            logits = outputs[
                "logits"
            ]

            loss = self.criterion(
                logits,
                labels
            )

            running_loss += (
                loss.item()
            )

            preds = logits.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                preds == labels
            ).sum().item()

        return {

            "loss":
                running_loss
                /
                len(dataloader),

            "accuracy":
                correct
                /
                total
        }
