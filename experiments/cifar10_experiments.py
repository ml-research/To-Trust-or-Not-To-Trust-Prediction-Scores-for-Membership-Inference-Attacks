import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
from torch.utils.data import Subset, dataset
import torchvision
import torchvision.transforms as T
import numpy as np
import csv
import argparse
from rtpt import RTPT

from attacks import SalemAttack, EntropyAttack, ThresholdAttack
from datasets import StanfordDogs, FakeCIFAR10, AFHQ
from utils.dataset_utils import get_subsampled_dataset, get_train_val_split, get_normalization, \
    create_permuted_dataset, create_scaled_dataset, create_un_normalized_dataset
from utils.validation import evaluate, expected_calibration_error, overconfidence_error
from experiment_utils import train, attack_model, write_results_to_csv, get_llla_calibrated_models, \
    get_temp_calibrated_models

from models.cifar10_models import ResNet18, SalemCNN_Relu, EfficientNetB0

# --------------------------------------
# ARGUMENT PARSER
# --------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', help='Whether to train the model')
parser.add_argument('--seed', default=42, type=int, help='The seed to use')
parser.add_argument(
    '--model_arch',
    default='resnet18',
    choices=['salem_cnn_relu', 'efficient_net', 'resnet18'],
    help='The model architecture to use'
)
parser.add_argument(
    '--epochs', default=100, type=int, help='The number of epochs the target and shadow model are trained for'
)
parser.add_argument(
    '--train_set_size',
    default=12500,
    type=int,
    help='The size of the training set for the target and the shadow model respectively'
)
parser.add_argument(
    '--test_set_size',
    default=5000,
    type=int,
    help='The size of the test set size for the target and the shadow model respectively'
)
parser.add_argument(
    '--batch_size', default=64, type=int, help='The batch size used for training the target and the shadow model'
)
parser.add_argument(
    '--model_input_image_size', default=32, type=int, help='The size of the images used to train the model'
)
parser.add_argument(
    '--attack_set_size',
    default=2500,
    type=int,
    help='The size of the member and non-member set respectively used for the membership inference attacks'
)
parser.add_argument(
    '--salem_k',
    default=3,
    type=int,
    help='The number of top prediction scores used for the Salem membership inference attack'
)
parser.add_argument(
    '--label_smoothing',
    dest='label_smoothing',
    action='store_true',
    help='Whether to use label smoothing during training'
)
parser.add_argument(
    '--label_smoothing_factor', default=0.0083, type=float, help='The smoothing factor for label smoothing'
)
parser.add_argument('--weight_decay', default=0, type=float, help='The weight decay value to use for training')
parser.add_argument('--llla', dest='llla', action='store_true', help='Whether to use llla as calibration method')
parser.add_argument(
    '--temp_scaling', dest='temp_scaling', action='store_true', help='Whether to use temperature scaling'
)
parser.add_argument('--temp_value', default=None, type=float, help='Set a temperature value by hand')

args = parser.parse_args()
if args.label_smoothing:
    print(f'Training model using label smoothing factor: {args.label_smoothing_factor}')
if args.weight_decay != 0:
    print(f'Using weight decay with value {args.weight_decay}')
if args.llla:
    print(f'Using LLLA for calibration')
if args.temp_scaling and args.temp_value is None:
    print('Using temperature scaling for calibration and searching for best temperature value')
elif args.temp_scaling and args.temp_value is not None:
    print(f'Using temperature scaling with temperature value of {args.temp_value}')
elif not args.temp_scaling and args.temp_value is not None:
    raise Exception('To use temperature scaling as calibration method use the flag `--temp_scaling`')

# --------------------------------------
# GLOBAL VARIABLES
# --------------------------------------

DATASET_NAME = 'cifar10'
# parameters for the target/shadow model
TRAIN_MODEL = args.train
SEED = args.seed
MODEL_ARCH = args.model_arch
title_addition = ''
if args.label_smoothing:
    title_addition = 'LS_{}_'.format(args.label_smoothing_factor)
elif args.weight_decay != 0:
    title_addition = 'L2_{}_'.format(args.weight_decay)
TARGET_MODEL_FILE = os.path.join(
    os.path.dirname(__file__), f'pretrained_models/{MODEL_ARCH}_{DATASET_NAME}_{title_addition}target.pt'
)
SHADOW_MODEL_FILE = os.path.join(
    os.path.dirname(__file__), f'pretrained_models/{MODEL_ARCH}_{DATASET_NAME}_{title_addition}shadow.pt'
)
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
EPOCHS = args.epochs
TRAIN_SET_SIZE = args.train_set_size
TEST_SET_SIZE = args.test_set_size
BATCH_SIZE = args.batch_size
MODEL_INPUT_IMAGE_SIZE = args.model_input_image_size
WEIGHT_DECAY = args.weight_decay

# parameters for the membership inference attack
ATTACK_SET_SIZE = args.attack_set_size
SALEM_K = args.salem_k

LABEL_SMOOTHING = args.label_smoothing
LABEL_SMOOTHING_FACTOR = args.label_smoothing_factor
USE_LLLA = args.llla
USE_TEMP = args.temp_scaling
TEMP_VALUE = args.temp_value

# set the seed and set pytorch to behave deterministically
torch.manual_seed(SEED)
if not args.label_smoothing:
    torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_member_non_member_split(train_set: dataset, test_set: dataset, split_size: int):
    """
    Takes the train and test set and returns a subset of each set with the given number of samples.
    """
    # get the member subset of the target training data that can be used for finding a threshold and attacking the model
    member = get_subsampled_dataset(train_set, split_size)
    # get the non-member subset of the target test data that can be used for finding a threshold and attacking the model
    non_member = get_subsampled_dataset(test_set, split_size)

    return member, non_member


def get_model_architecture(arch_name: str):
    """
    Returns the instantiated model.
    """
    # create the model architecture
    if arch_name == 'resnet18':
        model = ResNet18(num_classes=len(original_train_dataset.classes))
    elif arch_name == 'salem_cnn_relu':
        model = SalemCNN_Relu(num_classes=len(original_train_dataset.classes))
    elif arch_name == 'efficient_net':
        model = EfficientNetB0(num_classes=len(original_train_dataset.classes))
    else:
        raise Exception('Unknown model architecture given')

    # set the correct device for the architecture
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.device = device

    return model


if __name__ == '__main__':
    # get the transform for the datasets
    dataset_normalize = get_normalization(DATASET_NAME)
    dataset_transform = T.Compose(
        [T.Resize((MODEL_INPUT_IMAGE_SIZE, MODEL_INPUT_IMAGE_SIZE)), T.ToTensor(), dataset_normalize]
    )

    original_train_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(DATA_FOLDER, 'cifar10'), train=True, download=True, transform=dataset_transform
    )
    original_test_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(DATA_FOLDER, 'cifar10'), train=False, download=True, transform=dataset_transform
    )

    # get the cifar100 dataset as OOD dataset
    cifar100_ood_dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(DATA_FOLDER, 'cifar100'), train=True, download=True, transform=dataset_transform
    )

    # get the stl10 dataset and filter out all monkeys since there are no monkeys in the cifar10 dataset
    stl10_to_cifar = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: -1, 8: 8, 9: 9}

    def target_transform_stl(idx):
        return stl10_to_cifar[idx]

    stl10_ood_dataset = torchvision.datasets.STL10(
        root=os.path.join(DATA_FOLDER, 'stl10'),
        split='train',
        download=True,
        transform=dataset_transform,
        target_transform=target_transform_stl
    )
    #filter out all images with label 'monkey'
    stl10_ood_dataset = Subset(stl10_ood_dataset, np.where(stl10_ood_dataset.labels != 7)[0])

    # get the svhn dataset
    svhn_ood_dataset = torchvision.datasets.SVHN(
        root=os.path.join(DATA_FOLDER, 'svhn'), split='train', download=True, transform=dataset_transform
    )
    # get the stanford dogs dataset
    stanford_dogs_ood_dataset = StanfordDogs(root=os.path.join(DATA_FOLDER, 'stanford_dogs'), train=True, download=True, transform=dataset_transform)
    # get the generated cifar10 images
    fake_cifar_ood_dataset = FakeCIFAR10(root=os.path.join(DATA_FOLDER, 'fake_cifar10'), train=True, transform=dataset_transform)
    # get the afhq datasets and the subsets containing only cats and dogs respectively
    afhq_ood_dataset = AFHQ(root=os.path.join(DATA_FOLDER, 'afhq'), train=True, download=True, transform=dataset_transform)
    afhq_cats_ood_dataset = Subset(afhq_ood_dataset, np.where(np.array(afhq_ood_dataset.targets) == 0)[0])
    afhq_dogs_ood_dataset = Subset(afhq_ood_dataset, np.where(np.array(afhq_ood_dataset.targets) == 1)[0])

    # get the training dataset for the target/shadow model
    subsampled_train_dataset = get_subsampled_dataset(
        original_train_dataset, TRAIN_SET_SIZE * 2, seed=SEED, stratify=True
    )
    target_train, shadow_train = get_train_val_split(subsampled_train_dataset, 0.5, seed=SEED, stratify=True)
    # get the test dataset for the target/shadow model
    subsampled_test_dataset = original_test_dataset
    if len(original_test_dataset) > TEST_SET_SIZE * 2:
        subsampled_test_dataset = get_subsampled_dataset(
            original_test_dataset, TEST_SET_SIZE * 2, seed=SEED, stratify=False
        )
    target_test, shadow_test = get_train_val_split(subsampled_test_dataset, 0.5, seed=SEED, stratify=False)
    print('')
    print(f'Trainingset size: {len(subsampled_train_dataset)} - Testset size: {len(subsampled_test_dataset)}')
    print(f'Trainingset size target: {len(target_train)} - Testset size target: {len(target_test)}')
    print(f'Trainingset size shadow: {len(shadow_train)} - Testset size shadow: {len(shadow_test)}')

    # get the member and non-member for the target model
    member_target, non_member_target = get_member_non_member_split(target_train, target_test, ATTACK_SET_SIZE)
    print(f'Size Member Target: {len(member_target)} \t Size Non-Member Target: {len(non_member_target)}')
    # get the member and non-member for the shadow model
    member_shadow, non_member_shadow = get_member_non_member_split(shadow_train, shadow_test, ATTACK_SET_SIZE)
    print(f'Size Member Shadow: {len(member_shadow)} \t Size Non-Member Shadow: {len(non_member_shadow)}')

    # get the target and shadow model architecture
    target_model = get_model_architecture(MODEL_ARCH)
    shadow_model = get_model_architecture(MODEL_ARCH)

    # train or load the model
    if TRAIN_MODEL:
        rtpt = RTPT(name_initials='', experiment_name=f'{MODEL_ARCH.upper()}', max_iterations=EPOCHS * 2)
        rtpt.start()
        train(
            model=target_model,
            train_set=target_train,
            test_set=target_test,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            model_arch=MODEL_ARCH,
            filename=TARGET_MODEL_FILE,
            weight_decay=WEIGHT_DECAY,
            label_smoothing_factor=LABEL_SMOOTHING_FACTOR if LABEL_SMOOTHING else None,
            rtpt=rtpt
        )
        train(
            model=shadow_model,
            train_set=shadow_train,
            test_set=shadow_test,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            model_arch=MODEL_ARCH,
            filename=SHADOW_MODEL_FILE,
            weight_decay=WEIGHT_DECAY,
            label_smoothing_factor=LABEL_SMOOTHING_FACTOR if LABEL_SMOOTHING else None,
            rtpt=rtpt
        )
    else:
        target_model.load_state_dict(torch.load(TARGET_MODEL_FILE))
        shadow_model.load_state_dict(torch.load(SHADOW_MODEL_FILE))
    print('')
    print(
        f'Target Model Training Acc={evaluate(target_model, target_train):.4f} ' +
        f'\t Target Model Test Acc={evaluate(target_model, target_test):.4f}'
    )
    print(
        f'Shadow Model Training Acc={evaluate(shadow_model, shadow_train):.4f} ' +
        f'\t Shadow Model Test Acc={evaluate(shadow_model, shadow_test):.4f}'
    )
    print(f'ECE Target={expected_calibration_error(target_model, target_test, num_bins=15, apply_softmax=True):.4f}')
    print(f'ECE Shadow={expected_calibration_error(shadow_model, shadow_test, num_bins=15, apply_softmax=True):.4f}')
    print(f'Overconfidence Error Target={overconfidence_error(target_model, target_test, num_bins=15, apply_softmax=True):.4f}')
    print(f'Overconfidence Error Shadow={overconfidence_error(shadow_model, shadow_test, num_bins=15, apply_softmax=True):.4f}')

    # create the permuted image dataset
    permuted_non_member_target = create_permuted_dataset(non_member_target)
    # create the scaled image dataset
    scaled_non_member_target = create_scaled_dataset(non_member_target)
    # create the un-normalized image dataset
    un_normalized_non_member_target = create_un_normalized_dataset(non_member_target, DATASET_NAME)
    # take a subset from the ood datasets
    stl10_ood_non_member_target = get_subsampled_dataset(stl10_ood_dataset, ATTACK_SET_SIZE)
    cifar100_ood_non_member_target = get_subsampled_dataset(cifar100_ood_dataset, ATTACK_SET_SIZE)
    svhn_ood_non_member_target = get_subsampled_dataset(svhn_ood_dataset, ATTACK_SET_SIZE)
    stanford_dogs_ood_non_member_target = get_subsampled_dataset(stanford_dogs_ood_dataset, ATTACK_SET_SIZE)
    fake_cifar10_ood_non_member_target = get_subsampled_dataset(fake_cifar_ood_dataset, ATTACK_SET_SIZE)
    afhq_dogs_ood_non_member_target = get_subsampled_dataset(afhq_dogs_ood_dataset, ATTACK_SET_SIZE)
    afhq_cats_ood_non_member_target = get_subsampled_dataset(afhq_cats_ood_dataset, ATTACK_SET_SIZE)

    # if LLLA should be used calibrate the target and shadow model
    if USE_LLLA:
        target_model, shadow_model = get_llla_calibrated_models(
            target_model=target_model,
            shadow_model=shadow_model,
            non_member_target=non_member_target,
            non_member_shadow=non_member_shadow,
            target_train=target_train,
            shadow_train=shadow_train,
            attack_set_size=ATTACK_SET_SIZE,
            dataset_transform=dataset_transform,
            batch_size=BATCH_SIZE,
            image_size=32
        )
        print(
            f'ECE LLLA Calibrated Target={expected_calibration_error(target_model, target_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(
            f'ECE LLLA Calibrated Shadow={expected_calibration_error(shadow_model, shadow_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(f'LLLA Calibrated Target Model Test Acc={evaluate(target_model, target_test):.4f}')
        print(f'LLLA Calibrated Shadow Model Test Acc={evaluate(shadow_model, shadow_test):.4f}')
        print(f'Overconfidence Error LLLA Calibrated Target Model={overconfidence_error(target_model, target_test, num_bins=15, apply_softmax=False):.4f}')
        print(f'Overconfidence Error LLLA Calibrated Shadow Model={overconfidence_error(shadow_model, shadow_test, num_bins=15, apply_softmax=False):.4f}')

    # if temperature scaling should be used calibrate the target and shadow model
    if USE_TEMP:
        target_model, shadow_model = get_temp_calibrated_models(
            target_model=target_model,
            shadow_model=shadow_model,
            non_member_target=non_member_target,
            non_member_shadow=non_member_shadow,
            temp_value=TEMP_VALUE
        )
        print(
            f'ECE Temp. Calibrated Target={expected_calibration_error(target_model, target_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(
            f'ECE Temp. Calibrated Shadow={expected_calibration_error(shadow_model, shadow_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(f'Temp. Calibrated Target Model Test Acc={evaluate(target_model, target_test):.4f}')
        print(f'Temp. Calibrated Shadow Model Test Acc={evaluate(shadow_model, shadow_test):.4f}')
        print(f'Overconfidence Error Temp. Calibrated Target Model={overconfidence_error(target_model, target_test, num_bins=15, apply_softmax=False):.4f}')
        print(f'Overconfidence Error Temp. Calibrated Shadow Model={overconfidence_error(shadow_model, shadow_test, num_bins=15, apply_softmax=False):.4f}')

    # create the attacks
    attacks = [
        ThresholdAttack(apply_softmax=not (USE_LLLA or USE_TEMP)),
        SalemAttack(apply_softmax=not (USE_LLLA or USE_TEMP), k=SALEM_K),
        EntropyAttack(apply_softmax=not (USE_LLLA or USE_TEMP))
    ]
    # learn the attack parameters for each attack
    for attack in attacks:
        attack.learn_attack_parameters(shadow_model, member_shadow, non_member_shadow)

    # create the csv writer to write the attack results to a csv file
    title_addition = ''
    if LABEL_SMOOTHING:
        title_addition = 'LS_{}_'.format(LABEL_SMOOTHING_FACTOR)
    elif WEIGHT_DECAY != 0:
        title_addition = 'L2_{}_'.format(WEIGHT_DECAY)
    elif USE_LLLA:
        title_addition = 'LLLA_'
    elif USE_TEMP:
        title_addition = 'temp_' + (f'{TEMP_VALUE}_' if TEMP_VALUE is not None else '') + 'calib_'
    with open(
        os.path.join(RESULTS_FOLDER, f'{MODEL_ARCH}_{DATASET_NAME}_{title_addition}attack_results.csv'), 'w'
    ) as csv_file:
        # create the csv writer and write the labels for the columns
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        column_labels = ['']
        for attack in attacks:
            column_labels.extend(
                [
                    f'{attack.display_name} {metric}'
                    for metric in ['Precision', 'Recall', 'AUROC', 'AUPR', 'FPR@95%TPR', 'FPR', 'TP_MMPS', 'FP_MMPS', 'FN_MMPS', 'TN_MMPS']
                ]
            )
        csv_writer.writerow(column_labels)

        # attack the models using the different non-member sets
        print('')
        print('Attack Model using Original Non-Members:')
        results = attack_model(target_model, attacks, member_target, non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Original')

        print('\n')
        print('Attack Model using Permuted Non-Members:')
        results = attack_model(target_model, attacks, member_target, permuted_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Permuted')

        print('\n')
        print('Attack Model using Scaled Non-Members:')
        results = attack_model(target_model, attacks, member_target, scaled_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Scaled')

        print('\n')
        print('Attack Model using Non-Members without Normalization:')
        results = attack_model(target_model, attacks, member_target, un_normalized_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='No Normalization')

        print('\n')
        print('Attack Model using STL-10 OOD Non-Members:')
        results = attack_model(target_model, attacks, member_target, stl10_ood_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='STL-10')

        print('\n')
        print('Attack Model using CIFAR-100 OOD Non-Members:')
        results = attack_model(target_model, attacks, member_target, cifar100_ood_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='CIFAR100')

        print('\n')
        print('Attack Model using SVHN OOD Non-Members:')
        results = attack_model(target_model, attacks, member_target, svhn_ood_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='SVHN')

        print('\n')
        print('Attack Model using Stanford Dogs OOD Non-Members:')
        results = attack_model(target_model, attacks, member_target, stanford_dogs_ood_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Stanford Dogs')

        print('\n')
        print('Attack Model using Fake Cifar-10 OOD Non-Members:')
        results = attack_model(target_model, attacks, member_target, fake_cifar10_ood_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Fake CIFAR-10')

        print('\n')
        print('Attack Model using AFHQ Dogs OOD Non-Members:')
        results = attack_model(target_model, attacks, member_target, afhq_dogs_ood_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='AFHQ Dogs')

        print('\n')
        print('Attack Model using AFHQ Cats OOD Non-Members:')
        results = attack_model(target_model, attacks, member_target, afhq_cats_ood_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='AFHQ Cats')
