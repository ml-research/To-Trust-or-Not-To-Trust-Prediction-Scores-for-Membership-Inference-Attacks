import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from attacks.attack import PredictionScoreAttack
from utils.training import EarlyStopper


class SalemAttack(PredictionScoreAttack):
    def __init__(
        self,
        apply_softmax: bool,
        k: int = 3,
        attack_model: nn.Module = None,
        batch_size: int = 16,
        epochs: Optional[int] = None,
        lr: float = 0.01,
        log_training: bool = False,
    ):
        super().__init__('Salem Attack')
        self.k = k
        self.batch_size = batch_size
        if attack_model:
            self.attack_model = attack_model
        else:
            self.attack_model = nn.Sequential(nn.Linear(self.k, 64), nn.ReLU(), nn.Linear(64, 1))
        self.attack_model.to(self.device)
        self.apply_softmax = apply_softmax
        self.epochs = epochs
        self.lr = lr
        self.log_training = log_training

    def learn_attack_parameters(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset):
        # Gather predictions by shadow model
        shadow_model.to(self.device)
        shadow_model.eval()
        predictions = []
        membership_labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for i, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = shadow_model(x)
                    if self.apply_softmax:
                        prediction_scores = torch.softmax(output, dim=1)
                        predictions.append(prediction_scores)
                    else:
                        predictions.append(output)
                    membership_labels.append(torch.full_like(y, i))

        # Compute top-k predictions
        predictions = torch.cat(predictions, dim=0)
        membership_labels = torch.cat(membership_labels, dim=0)
        top_k_predictions = torch.topk(predictions, k=self.k, dim=1, largest=True, sorted=True).values
        attack_dataset = torch.utils.data.dataset.TensorDataset(top_k_predictions, membership_labels)

        # Train attack model
        self.attack_model.train()
        loss_fkt = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=self.lr)
        if self.log_training:
            print('Train attack model')

        early_stopper = EarlyStopper(window=15, min_diff=0.0005)
        epoch = 0
        while epoch != self.epochs:
            num_corrects = 0
            total_samples = 0
            running_loss = 0.0
            for x, y in DataLoader(attack_dataset, batch_size=self.batch_size, shuffle=True):
                x, y = x.to(self.device), y.to(self.device)
                output = self.attack_model(x).squeeze()
                loss = loss_fkt(output, y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = (output.sigmoid() >= 0.5).long().squeeze()
                num_corrects += torch.sum(preds == y.squeeze())
                total_samples += len(preds)
                running_loss += loss.item() * x.size(0)

            acc = num_corrects / total_samples
            train_loss = running_loss / len(attack_dataset)

            if self.log_training:
                print(f'Epoch {epoch}: Acc={acc:.4f}')

            if early_stopper.stop_early(train_loss):
                break

            epoch += 1

    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        predictions = []
        self.attack_model.eval()
        target_model.eval()
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=self.batch_size, num_workers=4):
                x, y = x.to(self.device), y.to(self.device)
                target_output = target_model(x)
                if self.apply_softmax:
                    pred_scores = torch.softmax(target_output, dim=1)
                    top_pred_scores = torch.topk(pred_scores, k=self.k, dim=1, largest=True, sorted=True).values
                    attack_output = self.attack_model(top_pred_scores)
                else:
                    top_pred_scores = torch.topk(target_output, k=self.k, dim=1, largest=True, sorted=True).values
                    attack_output = self.attack_model(top_pred_scores)
                predictions.append(attack_output.sigmoid() >= 0.5)
        predictions = torch.cat(predictions, dim=0).squeeze()
        return predictions.cpu().numpy()

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        predictions = []
        self.attack_model.eval()
        target_model.eval()
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=self.batch_size):
                x, y = x.to(self.device), y.to(self.device)
                target_output = target_model(x)
                if self.apply_softmax:
                    pred_scores = torch.softmax(target_output, dim=1)
                    top_pred_scores = torch.topk(pred_scores, k=self.k, dim=1, largest=True, sorted=True).values
                    attack_output = self.attack_model(top_pred_scores)
                else:
                    top_pred_scores = torch.topk(target_output, k=self.k, dim=1, largest=True, sorted=True).values
                    attack_output = self.attack_model(top_pred_scores)
                predictions.append(attack_output.sigmoid())
        return torch.cat(predictions, dim=0).squeeze().cpu()
