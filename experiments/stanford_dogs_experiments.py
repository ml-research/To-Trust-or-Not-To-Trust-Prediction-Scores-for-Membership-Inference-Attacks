import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as T
from torchvision.models.resnet import *
import numpy as np

import csv
from rtpt.rtpt import RTPT
import argparse

from attacks import SalemAttack, EntropyAttack, ThresholdAttack
from datasets import StanfordDogs, fake_dogs, AFHQ
from utils.dataset_utils import get_subsampled_dataset, get_train_val_split, create_permuted_dataset, \
    create_scaled_dataset, create_un_normalized_dataset, get_normalization
from utils.validation import evaluate, expected_calibration_error, overconfidence_error
from experiment_utils import train, attack_model, write_results_to_csv, get_llla_calibrated_models, \
    get_temp_calibrated_models

# --------------------------------------
# ARGUMENT PARSER
# --------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    '--train', dest='train', action='store_true', help='Whether to train the model or lead the pre-trained model'
)
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='Whether to use the model weights pre-trained on ImageNet'
)
parser.add_argument('--seed', default=42, type=int, help='The seed to use')
parser.add_argument(
    '--epochs', default=100, type=int, help='The number of epochs the target and shadow model are trained for'
)
parser.add_argument(
    '--train_set_size',
    default=8232,  # this is the number of samples when the Stanford Dogs dataset is downsampled to 40%
    type=int,
    help='The size of the training set for the target and the shadow model respectively'
)
parser.add_argument(
    '--test_set_size',
    default=2058,  # this is the number of samples when the Stanford Dogs dataset is downsampled to 10%,
    type=int,
    help='The size of the test set size for the target and the shadow model respectively'
)
parser.add_argument(
    '--batch_size', default=64, type=int, help='The batch size used for training the target and the shadow model'
)
parser.add_argument(
    '--model_input_image_size', default=224, type=int, help='The size of the images used to train the model'
)
parser.add_argument(
    '--attack_set_size',
    default=2058,
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
    '--label_smoothing_factor', default=0.1, type=float, help='The smoothing factor for label smoothing'
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

if args.pretrained and not args.train:
    raise Exception('Using model pre-trained on ImageNet can only be used when re-training the network')

# --------------------------------------
# GLOBAL VARIABLES
# --------------------------------------

# parameters for the target/shadow model
DATASET_NAME = 'stanford_dogs'
TRAIN_MODEL = args.train
SEED = args.seed
MODEL_ARCH = 'resnet50'  # 'resnet50', 'resnet101', 'resnet152'
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
IMAGENET_PRETRAINED = args.pretrained

# parameters for the membership inference attack
ATTACK_SET_SIZE = args.attack_set_size
SALEM_K = args.salem_k

WEIGHT_DECAY = args.weight_decay
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


def get_model_architecture(pretrained):
    model = resnet50(pretrained)
    model.fc = torch.nn.Linear(2048, 120, bias=True)
    # set the correct device for the architecture
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.device = device

    return model


if __name__ == '__main__':
    # get the transform for the datasets
    dataset_normalize = get_normalization(DATASET_NAME)
    transformation_dogs_train = T.Compose(
        [
            T.RandomRotation(degrees=20),
            T.Resize(230),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            dataset_normalize
        ]
    )

    transformation_without_augmentation = T.Compose(
        [T.Resize(224), T.CenterCrop((224, 224)), T.ToTensor(), dataset_normalize]
    )

    dataset_train = StanfordDogs(DATA_FOLDER, transform=transformation_dogs_train, all_data=True, download=True)
    dataset_train, _ = get_train_val_split(dataset_train, 0.8, seed=SEED)
    dataset_no_augmentation = StanfordDogs(DATA_FOLDER, transform=transformation_without_augmentation, all_data=True)
    dataset_train_no_augmentation, dataset_test = get_train_val_split(dataset_no_augmentation, 0.8, seed=SEED)

    # get the training dataset for the target/shadow model
    target_train, shadow_train = get_train_val_split(dataset_train, 0.5, seed=SEED)

    stl10_to_cifar = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: -1, 8: 8, 9: 9}

    def target_transform_stl(idx):
        return stl10_to_cifar[idx]

    stl10_ood_dataset = torchvision.datasets.STL10(
        root=DATA_FOLDER,
        split='train',
        download=True,
        transform=transformation_without_augmentation,
        target_transform=target_transform_stl
    )
    # Filter out all samples without label 'dog'
    stl10_ood_dataset = Subset(stl10_ood_dataset, np.where(stl10_ood_dataset.labels == 5)[0])

    afhq_dataset = AFHQ(DATA_FOLDER, transform=transformation_without_augmentation, download=True)
    afhq_dogs = Subset(afhq_dataset, np.where(np.array(afhq_dataset.targets) == 1)[0])
    afhq_cats = Subset(afhq_dataset, np.where(np.array(afhq_dataset.targets) == 0)[0])
    afhq_wilds = Subset(afhq_dataset, np.where(np.array(afhq_dataset.targets) == 2)[0])
    afhq_rest = Subset(afhq_dataset, np.where(np.array(afhq_dataset.targets) != 1)[0])
    fake_stanford_dogs = fake_dogs.FakeAFHQDogs(DATA_FOLDER, train=True, transform=transformation_without_augmentation)
    print('')
    print(f'Number fake dog samples: {len(fake_stanford_dogs)}')
    print(f'Trainingset size: {len(dataset_train)} - Testset size: {len(dataset_test)}')
    print(f'Trainingset size target: {len(target_train)} - Testset size target: {len(dataset_test)}')
    print(f'Trainingset size shadow: {len(shadow_train)} - Testset size shadow: {len(dataset_test)}')

    # get the members for the target and shadow model model
    dataset_train_no_augmentation_target, dataset_train_no_augmentation_shadow = get_train_val_split(dataset_train_no_augmentation, 0.5, seed=SEED)
    # get the non-members for the target and shadow model model
    non_member_target, non_member_shadow = get_train_val_split(dataset_test, 0.5, seed=SEED)
    member_target = get_subsampled_dataset(
        dataset_train_no_augmentation_target, dataset_size=len(non_member_target), seed=SEED
    )
    member_shadow = get_subsampled_dataset(
        dataset_train_no_augmentation_shadow, dataset_size=len(non_member_shadow), seed=SEED
    )
    print('')
    print(f'Size Member Target: {len(member_target)} \t Size Non-Member Target: {len(non_member_target)}')
    print(f'Size Member Shadow: {len(member_shadow)} \t Size Non-Member Shadow: {len(non_member_shadow)}')

    # get the target and shadow model architecture
    target_model = get_model_architecture(IMAGENET_PRETRAINED)
    shadow_model = get_model_architecture(IMAGENET_PRETRAINED)

    # train or load the model
    if TRAIN_MODEL:
        rtpt = RTPT(name_initials='', experiment_name=f'{MODEL_ARCH.upper()}', max_iterations=EPOCHS * 2)
        rtpt.start()
        train(
            model=target_model,
            train_set=target_train,
            test_set=dataset_test,
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
            test_set=dataset_test,
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
        f'Target Model Training Acc={evaluate(target_model, dataset_train_no_augmentation_target):.4f} ' +
        f'\t Target Model Test Acc={evaluate(target_model, dataset_test):.4f}'
    )
    print(
        f'Shadow Model Training Acc={evaluate(shadow_model, dataset_train_no_augmentation_shadow):.4f} ' +
        f'\t Shadow Model Test Acc={evaluate(shadow_model, dataset_test):.4f}'
    )
    print(f'ECE Target={expected_calibration_error(target_model, dataset_test, num_bins=15, apply_softmax=True):.4f}')
    print(f'ECE Shadow={expected_calibration_error(shadow_model, dataset_test, num_bins=15, apply_softmax=True):.4f}')
    print(f'Overconfidence Error Target={overconfidence_error(target_model, dataset_test, num_bins=15, apply_softmax=True):.4f}')
    print(f'Overconfidence Error Shadow={overconfidence_error(shadow_model, dataset_test, num_bins=15, apply_softmax=True):.4f}')

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
            dataset_transform=transformation_without_augmentation,
            batch_size=BATCH_SIZE,
            image_size=224
        )
        print(
            f'ECE LLLA Calibrated Target={expected_calibration_error(target_model, dataset_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(
            f'ECE LLLA Calibrated Shadow={expected_calibration_error(shadow_model, dataset_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(f'LLLA Calibrated Target Model Test Acc={evaluate(target_model, dataset_test):.4f}')
        print(f'LLLA Calibrated Shadow Model Test Acc={evaluate(shadow_model, dataset_test):.4f}')

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
            f'ECE Temp. Calibrated Target={expected_calibration_error(target_model, dataset_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(
            f'ECE Temp. Calibrated Shadow={expected_calibration_error(shadow_model, dataset_test, num_bins=15, apply_softmax=False):.4f}'
        )
        print(f'Temp. Calibrated Target Model Test Acc={evaluate(target_model, dataset_test):.4f}')
        print(f'Temp. Calibrated Shadow Model Test Acc={evaluate(shadow_model, dataset_test):.4f}')

    # create the attacks
    attacks = [
        ThresholdAttack(apply_softmax=not (USE_LLLA or USE_TEMP)),
        SalemAttack(apply_softmax=not (USE_LLLA or USE_TEMP), k=SALEM_K),
        EntropyAttack(apply_softmax=not (USE_LLLA or USE_TEMP))
    ]
    # learn the attack parameters for each attack
    for attack in attacks:
        attack.learn_attack_parameters(shadow_model, member_shadow, non_member_shadow)

    # create the permuted image dataset
    permuted_non_member_target = create_permuted_dataset(non_member_target)
    # create the scaled image dataset
    scaled_non_member_target = create_scaled_dataset(non_member_target)
    # create the un-normalized image dataset
    un_normalized_non_member_target = create_un_normalized_dataset(non_member_target, DATASET_NAME)

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
        os.path.join(RESULTS_FOLDER, f'{MODEL_ARCH}_{DATASET_NAME}_{title_addition}performance_results.csv'), 'w'
    ) as csv_file:
        # create the csv writer and write the labels for the columns
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        column_labels = ['']
        for attack in attacks:
            column_labels.extend([f'{attack.display_name} {metric}' for metric in ['Precision', 'Recall', 'FPR']])
        csv_writer.writerow(column_labels)

        # compute train and test accuracy
        print('Evaluating prediction accuracy on target model')
        train_acc_target = evaluate(target_model, dataset_train_no_augmentation_target)
        csv_writer.writerow(['Train Acc. Target'] + [f'{train_acc_target:.4f}'] + [''] * 8)
        test_acc_target = evaluate(target_model, dataset_test)
        csv_writer.writerow(['Test Acc. Target'] + [f'{test_acc_target:.4f}'] + [''] * 8)
        ece_target = expected_calibration_error(
            target_model, dataset_test, num_bins=15, apply_softmax=not (USE_LLLA or USE_TEMP)
        )
        csv_writer.writerow(['ECE Target'] + [f'{ece_target:.4f}'] + [''] * 8)

        print('Evaluating prediction accuracy on shadow model')
        train_acc_shadow = evaluate(shadow_model, dataset_train_no_augmentation_shadow)
        csv_writer.writerow(['Train Acc. Shadow'] + [f'{train_acc_shadow:.4f}'] + [''] * 8)
        test_acc_shadow = evaluate(shadow_model, dataset_test)
        csv_writer.writerow(['Test Acc. Shadow'] + [f'{test_acc_shadow:.4f}'] + [''] * 8)
        ece_shadow = expected_calibration_error(
            shadow_model, dataset_test, num_bins=15, apply_softmax=not (USE_LLLA or USE_TEMP)
        )
        csv_writer.writerow(['ECE Target'] + [f'{ece_shadow:.4f}'] + [''] * 8)

    # create the csv writer to write the attack results to a csv file
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
                    for metric in ['Precision', 'Recall', 'FPR', 'TP_MMPS', 'FP_MMPS', 'FN_MMPS', 'TN_MMPS']
                ]
            )
        csv_writer.writerow(column_labels)

        # attack the models using the different non-member sets
        print('Attack Un-Calibrated Model using Original Non-Members:')
        results = attack_model(target_model, attacks, member_target, non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Original')

        print('\n')
        print('Attack Un-Calibrated Model using Fake Dogs Non-Members:')
        results = attack_model(target_model, attacks, member_target, fake_stanford_dogs)
        write_results_to_csv(csv_writer, results, row_label='Fake Dogs')

        print('\n')
        print('Attack Un-Calibrated Model using AFHQ Dogs Non-Members:')
        results = attack_model(target_model, attacks, member_target, afhq_dogs)
        write_results_to_csv(csv_writer, results, row_label='AFHQ-Dogs')

        print('\n')
        print('Attack Un-Calibrated Model using AFHQ Cats Non-Members:')
        results = attack_model(target_model, attacks, member_target, afhq_cats)
        write_results_to_csv(csv_writer, results, row_label='AFHQ-Cats')

        print('\n')
        print('Attack Un-Calibrated Model using AFHQ Wild Non-Members:')
        results = attack_model(target_model, attacks, member_target, afhq_wilds)
        write_results_to_csv(csv_writer, results, row_label='AFHQ-Wilds')

        print('\n')
        print('Attack Un-Calibrated Model using AFHQ Non-Dogs Non-Members:')
        results = attack_model(target_model, attacks, member_target, afhq_rest)
        write_results_to_csv(csv_writer, results, row_label='AFHQ-NonDogs')

        print('\n')
        print('Attack Un-Calibrated Model using Permuted Non-Members:')
        results = attack_model(target_model, attacks, member_target, permuted_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Permuted')

        print('\n')
        print('Attack Un-Calibrated Model using Scaled Non-Members:')
        results = attack_model(target_model, attacks, member_target, scaled_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='Scaled')

        print('\n')
        print('Attack Un-Calibrated Model using Non-Members without Normalization:')
        results = attack_model(target_model, attacks, member_target, un_normalized_non_member_target)
        write_results_to_csv(csv_writer, results, row_label='No Normalization')
