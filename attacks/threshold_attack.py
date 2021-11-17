import torch
import torch.nn as nn
from torch.utils.data import Dataset

from attacks.attack import PredictionScoreAttack
from utils.model_utils import get_model_prediction_scores
from utils.roc import get_roc


class ThresholdAttack(PredictionScoreAttack):
    def __init__(self, apply_softmax: bool):
        super().__init__('Threshold Attack')
        self.apply_softmax = apply_softmax
        self.attack_treshold = 0.0

    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset, *kwargs
    ):
        labels = [0 for _ in range(len(non_member_dataset))] + [1 for _ in range(len(member_dataset))]

        # get the prediction scores of the shadow model on the members and the non-members in order to attack the target model
        pred_scores_shadow_member = get_model_prediction_scores(
            model=shadow_model, apply_softmax=self.apply_softmax, dataset=member_dataset, batch_size=512, num_workers=8
        ).max(dim=1)[0].tolist()
        pred_scores_shadow_non_member = get_model_prediction_scores(
            model=shadow_model,
            apply_softmax=self.apply_softmax,
            dataset=non_member_dataset,
            batch_size=512,
            num_workers=8
        ).max(dim=1)[0].tolist()

        pred_scores = pred_scores_shadow_non_member + pred_scores_shadow_member
        self.shadow_fpr, self.shadow_tpr, self.thresholds, self.auroc = get_roc(labels, pred_scores)
        threshold_idx = (self.shadow_tpr - self.shadow_fpr).argmax()
        self.attack_treshold = self.thresholds[threshold_idx]

    def predict_membership(self, model: nn.Module, dataset: Dataset):
        # get the prediction scores of the shadow model on the members and the non-members in order to attack the target model
        pred_scores = get_model_prediction_scores(
            model=model, apply_softmax=self.apply_softmax, dataset=dataset, batch_size=512, num_workers=8
        ).max(dim=1)[0].tolist()

        return pred_scores > self.attack_treshold

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        return get_model_prediction_scores(
            model=target_model, apply_softmax=self.apply_softmax, dataset=dataset, batch_size=512, num_workers=8
        ).max(dim=1)[0]
