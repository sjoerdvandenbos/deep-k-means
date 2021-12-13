from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from learning_super import Learner
from modules.autoencoders import get_autoencoder


class SupervisedLearner(Learner):

    def __init__(self, args, specs):
        super().__init__(args, specs)

    def run_repeated_learning(self):
        for _ in range(self.n_runs):
            self._log(f"Run {self.run}")
            self.autoencoder = get_autoencoder(
                self.autoencoder_name,
                self.img_height,
                self.img_width,
                self.embedding_size,
                self.n_channels,
            ).to(self.device)
            if self.seeded:
                torch.manual_seed(self.seeds[self.run])
                np.random.seed(self.seeds[self.run])
            self.train()
            self.run += 1
        duration = datetime.now() - self.start_time
        self._log(f"Learning duration: {duration}")

    def train(self):
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), self.lr)
        for epoch in range(self.n_pretrain_epochs):
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            predictions = torch.zeros((self.trainset.target.shape[0], self.embedding_size), dtype=torch.float)
            train_losses = []
            self._log(f"Training step: epoch {epoch}")
            self.autoencoder.train()
            for train_indices, train_batch, train_labels in train_loader:
                train_batch = train_batch.to(self.device)

                # Train one mini-batch
                self.autoencoder.zero_grad(set_to_none=True)
                logits = self.autoencoder.forward_encoder(train_batch)
                prediction = F.softmax(logits, dim=1)
                target = self._label_to_one_hot(train_labels)
                loss = F.mse_loss(prediction, target)
                loss.backward()
                optimizer.step()

                # Save metrics to cpu memory
                predictions[train_indices, :] = prediction.detach().cpu()
                train_losses.append(loss.item())

                del logits, prediction, target, loss

            self._log_accuracy(predictions, self.trainset.target, "Train")
            self._log(f"Train losses: {sum(train_losses) / self.n_batches}")

            test_loader = DataLoader(self.testset, batch_size=self.test_size, shuffle=True, pin_memory=True)
            test_predictions = torch.zeros((len(self.testset), self.embedding_size), dtype=torch.float)
            test_losses = []
            self.autoencoder.eval()
            with torch.no_grad():
                for test_indices, test_batch, test_labels in test_loader:
                    test_batch = test_batch.to(self.device)

                    # Eval one mini-batch
                    logits = self.autoencoder.forward_encoder(test_batch)
                    prediction = F.softmax(logits, dim=1)
                    target = self._label_to_one_hot(test_labels)
                    loss = F.mse_loss(prediction, target)

                    # Save metrics to cpu memory
                    test_predictions[test_indices, :] = prediction.detach().cpu()
                    test_losses.append(loss.item())

            self._log_accuracy(test_predictions, self.testset.target, "Test")
            self._log(f"Test losses: {sum(test_losses) / self.n_test_batches}")
            self._log("----------------------------------------------\n")

    def _label_to_one_hot(self, labels):
        one_hot = torch.zeros(size=(labels.shape[0], self.embedding_size), dtype=torch.float, device=self.device)
        one_hot[torch.arange(labels.shape[0]), labels.long()] = 1
        return one_hot

    def _log_accuracy(self, predictions, targets, phase):
        predicted_label = predictions.argmax(axis=1)
        acc = (predicted_label == targets).long().sum() / targets.shape[0]
        self._log(f"{phase} ACC: {acc}")
