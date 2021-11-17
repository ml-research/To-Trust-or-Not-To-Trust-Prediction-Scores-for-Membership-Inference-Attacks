from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
from torchmetrics.functional import roc, auroc, precision_recall_curve, auc

from utils.model_utils import get_model_prediction_scores


class AttackResult:
    attack_acc: float
    precision: float
    recall: float
    tpr: float
    tnr: float
    fpr: float
    fnr: float
    tp_mmps: float
    fp_mmps: float
    fn_mmps: float
    tn_mmps: float

    def __init__(
        self,
        attack_acc: float,
        precision: float,
        recall: float,
        auroc: float,
        aupr: float,
        fpr_at_tpr95: float,
        tpr: float,
        tnr: float,
        fpr: float,
        fnr: float,
        tp_mmps: float,
        fp_mmps: float,
        fn_mmps: float,
        tn_mmps: float
    ):
        self.attack_acc = attack_acc
        self.precision = precision
        self.recall = recall
        self.auroc = auroc
        self.aupr = aupr
        self.fpr_at_tpr95 = fpr_at_tpr95
        self.tpr = tpr
        self.tnr = tnr
        self.fpr = fpr
        self.fnr = fnr
        self.tp_mmps = tp_mmps
        self.fp_mmps = fp_mmps
        self.fn_mmps = fn_mmps
        self.tn_mmps = tn_mmps


class PredictionScoreAttack:
    """
    The base class for all score-based membership inference attacks.
    """
    def __init__(self, display_name: str = '', *kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.display_name = display_name

    @abstractmethod
    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset
    ):
        raise NotImplementedError('This function has to be implemented in a subclass')

    @abstractmethod
    def predict_membership(self, target_model: nn.Module, dataset: Dataset) -> np.ndarray:
        """
        Predicts whether the samples in the given dataset are a member of the given model.
        :param target_model: The given target model to predict the membership.
        :param dataset: The dataset that is going to be used to predict the membership.
        :returns: A numpy array containing bool values for each sample indicating whether the samples was is a member or not.
        """
        raise NotImplementedError('This function has to be implemented in a subclass')

    def evaluate(self, target_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset) -> AttackResult:
        """
        Evaluates the attack by predicting the membership for the member dataset as well as the non-member dataset.
        Returns a `AttackResult`-object.
        :param target_model: The given target model
        :param member_dataset: The member dataset that was used to train the target model.
        :param non_member_dataset: The non-member dataset that was **not** used to train the target model.
        :param kwargs: Additional optional parameters for the `predict_membership`-method.
        """
        member_predictions = self.predict_membership(target_model, member_dataset)
        non_member_predictions = self.predict_membership(target_model, non_member_dataset)
        tp = member_predictions.sum()
        tn = len(non_member_dataset) - non_member_predictions.sum()
        fp = non_member_predictions.sum()
        fn = len(member_dataset) - member_predictions.sum()
        tpr = tp / len(member_dataset)
        tnr = tn / len(non_member_dataset)
        fpr = 1 - tnr
        fnr = 1 - tpr
        pre = tp / (tp + fp) if tp != 0 else 0.0
        rec = tp / (tp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)

        member_pred_scores = self.get_attack_model_prediction_scores(target_model, dataset=member_dataset)
        non_member_pred_scores = self.get_attack_model_prediction_scores(target_model, dataset=non_member_dataset)
        concat_preds = torch.cat((non_member_pred_scores, member_pred_scores))
        concat_targets = torch.tensor([0 for _ in non_member_pred_scores] + [1 for _ in member_pred_scores])

        # get the auroc, aupr and FPR@95%TPR
        auroc_value: float = auroc(
            preds=concat_preds,
            target=concat_targets
        ).item()
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            preds=concat_preds,
            target=concat_targets
        )
        aupr_value: float = auc(x=pr_recall, y=pr_precision, reorder=True).item()
        tm_fpr, tm_tpr, tm_thresholds = roc(preds=concat_preds, target=concat_targets)
        tpr_greater95_indices = np.where(tm_tpr >= 0.95)[0]
        fpr_at_tpr95 = tm_fpr[tpr_greater95_indices[0]].item()

        # get the mmps values
        tp_pred_scores = self.get_pred_score_classified_as_members(
            target_model, member_dataset, apply_softmax=self.apply_softmax
        )
        tp_mmps = tp_pred_scores.max(dim=1)[0].mean() if len(tp_pred_scores) > 0 else 0

        fp_pred_scores = self.get_pred_score_classified_as_members(
            target_model, non_member_dataset, apply_softmax=self.apply_softmax
        )
        fp_mmps = fp_pred_scores.max(dim=1)[0].mean() if len(fp_pred_scores) > 0 else 0

        fn_pred_scores = self.get_pred_score_classified_as_non_members(
            target_model, member_dataset, apply_softmax=self.apply_softmax
        )
        fn_mmps = fn_pred_scores.max(dim=1)[0].mean() if len(fn_pred_scores) > 0 else 0

        tn_pred_scores = self.get_pred_score_classified_as_non_members(
            target_model, non_member_dataset, apply_softmax=self.apply_softmax
        )
        tn_mmps = tn_pred_scores.max(dim=1)[0].mean() if len(tn_pred_scores) > 0 else 0

        result = AttackResult(
            attack_acc=acc,
            precision=pre,
            recall=rec,
            auroc=auroc_value,
            aupr=aupr_value,
            fpr_at_tpr95=fpr_at_tpr95,
            tpr=tpr,
            tnr=tnr,
            fpr=fpr,
            fnr=fnr,
            tp_mmps=tp_mmps,
            fp_mmps=fp_mmps,
            fn_mmps=fn_mmps,
            tn_mmps=tn_mmps
        )
        return result

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        raise NotImplementedError('This function has to be implemented in a subclass')

    def get_pred_score_classified_as_members(self, target_model: nn.Module, dataset: Dataset, apply_softmax: bool):
        membership_predictions = self.predict_membership(target_model, dataset)

        predicted_member_indices = torch.nonzero(torch.tensor(membership_predictions).squeeze()).squeeze()
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)

        samples, labels = [], []
        for idx in predicted_member_indices:
            sample, label = dataset[idx]
            samples.append(sample)
            labels.append(label)
        samples = torch.stack(samples) if len(samples) > 0 else torch.empty(0)
        labels = torch.tensor(labels) if len(labels) > 0 else torch.empty(0)

        return get_model_prediction_scores(target_model, apply_softmax, torch.utils.data.TensorDataset(samples, labels))

    def get_pred_score_classified_as_non_members(self, target_model: nn.Module, dataset: Dataset, apply_softmax: bool):
        membership_predictions = self.predict_membership(target_model, dataset)

        predicted_member_indices = torch.nonzero(torch.logical_not(torch.tensor(membership_predictions)).squeeze()
                                                 ).squeeze()
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)

        samples, labels = [], []
        for idx in predicted_member_indices:
            sample, label = dataset[idx]
            samples.append(sample)
            labels.append(label)
        samples = torch.stack(samples) if len(samples) > 0 else torch.empty(0)
        labels = torch.tensor(labels) if len(labels) > 0 else torch.empty(0)

        return get_model_prediction_scores(target_model, apply_softmax, torch.utils.data.TensorDataset(samples, labels))
