import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import dataset

from rtpt import RTPT
from typing import Optional, List

from utils.training import train_model, LabelSmoothingCrossEntropy
from utils.dataset_utils import get_subsampled_dataset
from attacks import AttackResult, PredictionScoreAttack
from calibration import LLLA, TemperatureScaling


def train(
    model: nn.Module,
    train_set: dataset,
    test_set: dataset,
    batch_size: int,
    epochs: int,
    model_arch: str,
    filename: str,
    weight_decay: float,
    label_smoothing_factor: Optional[float],
    rtpt: RTPT = None
):
    """
    Trains the given model with the given parameters.
    """
    # train the target model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001 if model_arch == 'salem_cnn_relu' else 0.01, weight_decay=weight_decay
    )
    loss_fkt = nn.CrossEntropyLoss()
    if label_smoothing_factor is not None:
        loss_fkt = LabelSmoothingCrossEntropy(smoothing=label_smoothing_factor)

    train_model(
        model,
        train_set,
        optimizer,
        epochs,
        batch_size,
        loss_fkt,
        val_dataset=test_set,
        filename=filename,
        num_workers=16,
        rtpt=rtpt
    )


def print_attack_results(attack_name: str, attack_results: AttackResult):
    """
    Takes the attack name and the attack result object and prints the results to the console.
    """
    print(
        f'{attack_name}: \n ' + f'\tRecall: {attack_results.recall:.4f} \t Precision: {attack_results.precision:.4f} ' +
        f'\t AUROC: {attack_results.auroc:.4f} \t AUPR: {attack_results.aupr:.4f} \t FPR@95%TPR: {attack_results.fpr_at_tpr95:.4f}' +
        f'\t FPR: {attack_results.fpr:.4f} \t TP MMPS: {attack_results.tp_mmps:.4f} ' +
        f'\t FP MMPS: {attack_results.fp_mmps:.4f} \t FN MMPS: {attack_results.fn_mmps:.4f} ' +
        f'\t TN MMPS: {attack_results.tn_mmps:.4f}'
    )


def attack_model(model: nn.Module, attack_list: List[PredictionScoreAttack], member: dataset, non_member: dataset):
    """
    Takes the model, the list of attacks, a member and a non-member set and attacks the model with the given attacks.
    """
    attack_result_list: List[AttackResult] = []
    for current_attack in attack_list:
        attack_result = current_attack.evaluate(model, member, non_member)
        attack_result_list.append(attack_result)
        print_attack_results(current_attack.display_name, attack_result)
    return attack_result_list


def write_results_to_csv(csv_write, attack_results: List[AttackResult], row_label: str = ''):
    """
    Takes the csv writer, the results of the attacks and the row label and writes the results to the csv file.
    """
    columns: List[str] = [row_label]
    for current_result in attack_results:
        columns.extend(
            [
                f'{current_result.precision:.4f}',
                f'{current_result.recall:.4f}',
                f'{current_result.auroc:.4f}',
                f'{current_result.aupr:.4f}',
                f'{current_result.fpr_at_tpr95:.4f}',
                f'{current_result.fpr:.4f}',
                f'{current_result.tp_mmps:.4f}',
                f'{current_result.fp_mmps:.4f}',
                f'{current_result.fn_mmps:.4f}',
                f'{current_result.tn_mmps:.4f}'
            ]
        )
    # write the recall, the precision and the fpr to the next row
    csv_write.writerow(columns)


def get_llla_calibrated_models(
    target_model,
    shadow_model,
    non_member_target,
    non_member_shadow,
    target_train,
    shadow_train,
    attack_set_size,
    dataset_transform,
    batch_size,
    image_size=32
):
    # calibrate the target and shadow model together to prevent similar samples used for calibration
    calib_set_target = torchvision.datasets.FakeData(
        size=attack_set_size, image_size=(3, image_size, image_size), transform=dataset_transform, target_transform=lambda x: int(x)
    )
    calib_set_shadow = torchvision.datasets.FakeData(
        size=attack_set_size, image_size=(3, image_size, image_size), transform=dataset_transform, target_transform=lambda x: int(x)
    )

    # calib_dataset has to have the same length as the test loader of the target and the shadow model
    calib_set_target = get_subsampled_dataset(calib_set_target, attack_set_size)
    calib_set_shadow = get_subsampled_dataset(calib_set_shadow, attack_set_size)
    calib_dataset_loader_target = torch.utils.data.DataLoader(
        calib_set_target, batch_size=batch_size * 2, num_workers=8
    )
    calib_dataset_loader_shadow = torch.utils.data.DataLoader(
        calib_set_shadow, batch_size=batch_size * 2, num_workers=8
    )

    # assert that the calibration dataset, and the non-member sets have the same length.
    # This is important for LLA to work.
    assert len(calib_set_target) == len(calib_set_shadow)
    assert len(calib_set_target) == len(non_member_target)
    assert len(calib_set_shadow) == len(non_member_shadow)

    # create the train and test loader for calibration
    train_loader_target = torch.utils.data.DataLoader(target_train, batch_size=batch_size * 2, num_workers=8)
    train_loader_shadow = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size * 2, num_workers=8)
    test_loader_target = torch.utils.data.DataLoader(non_member_target, batch_size=batch_size * 2, num_workers=8)
    test_loader_shadow = torch.utils.data.DataLoader(non_member_shadow, batch_size=batch_size * 2, num_workers=8)

    # calibrate the target and the shadow model using the train, test and calibration loader
    target_model_llla_calibrated = LLLA(target_model, list(target_model.children())[-1])
    target_model_llla_calibrated.estimate_parameters(
        train_loader_target, test_loader_target, calib_dataset_loader_target, interval=torch.logspace(0, 3, 100)
    )
    shadow_model_llla_calibrated = LLLA(shadow_model, list(shadow_model.children())[-1])
    shadow_model_llla_calibrated.estimate_parameters(
        train_loader_shadow, test_loader_shadow, calib_dataset_loader_shadow, interval=torch.logspace(0, 3, 100)
    )

    return target_model_llla_calibrated, shadow_model_llla_calibrated


def get_temp_calibrated_models(
    target_model: nn.Module,
    shadow_model: nn.Module,
    non_member_target: dataset,
    non_member_shadow: dataset,
    temp_value: Optional[float] = None
):
    target_model_temp_calibrated = TemperatureScaling(target_model)
    shadow_model_temp_calibrated = TemperatureScaling(shadow_model)
    if temp_value is None:
        target_model_temp_calibrated.calibrate(non_member_target)
        shadow_model_temp_calibrated.calibrate(non_member_shadow)
    else:
        target_model_temp_calibrated.temperature = temp_value
        shadow_model_temp_calibrated.temperature = temp_value

    return target_model_temp_calibrated, shadow_model_temp_calibrated
