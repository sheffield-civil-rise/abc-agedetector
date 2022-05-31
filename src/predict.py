import argparse
import os
import datetime
import time
import json
import re

import tqdm  # progress bar


import torch
import torchvision
import pandas as pd

from torch.utils.data import DataLoader

from utils import is_valid_file
from dataset import generate_predict_dataset
from transforms import DefaultTransformation

from model import AgeDetector, BaseConvNet

_CLASS_DICT = {
    0: 'HISTORIC',
    1: 'INTERWAR',
    2: 'MODERN',
    3: 'POSTWAR',
    4: 'SIXTIES SEVENTIES'}
_LOG_STATES = {'DEBUG': 0, 'VERBOSE': 1, 'STATUS':2, 'QUIET': 3}

global LOG_STATE
LOG_STATE = _LOG_STATES['STATUS']

def _log(level, text):
    ''' logging function '''
    global LOG_STATE
    if type(level) is str:
        level = _LOG_STATES[level]
    if level < LOG_STATE:
        return
    if level == _LOG_STATES['DEBUG']:
        print('[d: ---')
        print(text)
        print('--]')
        input('↵ to continue')
    else:
        print(text)


def get_backbone(args):
    ''' get torch backbone based on input [default resnet18]'''
    backbone = 'resnet18' if args.backbone is None else args.backbone
    _log('VERBOSE', f'Using backbone {backbone}')

    return {
        'default': BaseConvNet, # i wrote this
        'resnet18': torchvision.models.resnet18,
        'alexnet': torchvision.models.alexnet,
        'inception_v3': torchvision.models.inception_v3,
        'mobilenet_v2': torchvision.models.mobilenet_v2}[backbone]


def prepare_dataset_loader(args):
    ''' '''
    _log('VERBOSE', 'Setting up dataset for prediction')
    transforms = DefaultTransformation(
        args.nb_patches, args.patch_size, args.crop_size)

    image_folder = generate_predict_dataset(
        args.base_dir,
        is_valid_file=is_valid_file,
        transforms=transforms)

    predict_loader = DataLoader(
        image_folder,
        batch_size=args.batch_size)
    return predict_loader


def create_model(args, nb_classes):
    ''' create model and load any weights if required '''
    backbone = get_backbone(args)

    _log('VERBOSE', f'Creating model with {nb_classes} classes')
    model = AgeDetector(
        nb_classes,
        backbone=backbone,
        pretrained=False,
        fixbackbone=False)

    _log('VERBOSE', f'Loading pretrained weights from {args.weights}')
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)

    model.eval()

    return model


def evaluate_model(model, predict_loader, device):
    ''' evaluate model on dataset '''
    since = time.time()

    _log('VERBOSE', f'Moving model to device:{device}')
    # load model to device
    model.to(device)
    model.eval()

    predictions = {}

    with torch.no_grad():
        with tqdm.tqdm(predict_loader, unit='batch') as tepoch:
            for inputs, fnames in tepoch:
                inputs = inputs.to(device)

                output = model(inputs)
                _, predicted = torch.max(output.data, 1)

                for idx, fname in enumerate(fnames):
                    _, _fname = os.path.split(fname)
                    predictions[_fname] = (
                        predicted[idx].cpu().item(),
                        torch.nn.functional.softmax(
                            output.data[idx,:], dim=0).cpu().numpy().tolist())

    elapsed = time.time() - since

    _log(
        'STATUS',
        f'Prediction complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')

    return predictions


def save_classes(classes, args):
    ''' write out classes to file '''
    _log('VERBOSE', f'writing classes to {args.out_path}')

    if args.as_csv:
        columns = ['path', 'class', 'class_id', 'scores']
        df = pd.DataFrame(columns=columns)
        for k, v in classes.items():
            df = pd.concat([df, pd.DataFrame(
                data=[[k, _CLASS_DICT[v[0]], v[0], v[1]]],
                columns=columns)], ignore_index=True)
        df.to_csv(args.out_path)
    else:
        with open(args.out_path, 'w') as fid:
            json.dump(classes, fid)


def main(args):
    ''' main body '''
    predict_loader = prepare_dataset_loader(args)

    model = create_model(args, args.nb_classes)

    # generate device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    _log('STATUS', f'Using device {device}')
    _log('DEBUG', 'Is this right ?')

    classes = evaluate_model(model, predict_loader, device)

    save_classes(classes, args)



def generate_parser():
    ''' Generates argument parser for commandline use '''
    # create parser object
    parser = argparse.ArgumentParser(
        description='Predicts age of buildings using trained model')
    parser.add_argument(
        'base_dir', type=str,
        help='folder containing images to categorise age for')
    parser.add_argument(
        'out_path', type=str,
        help='csv/json file to save ages to')
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to weights of pretrained model [required]')
    parser.add_argument(
        '-n', '--nb_classes', default=5,
        help='number of classes')
    parser.add_argument(
        '--fix_hyperparams', action='store_true',
        help='use set hyperparameters rather than inferring from file')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=1,
        help='batch size for prediction')
    parser.add_argument(
        '-p', '--nb_patches', type=int, default=None,
        help='[parameter] number of patches to extract from images')
    parser.add_argument(
        '-s', '--patch_size', type=int, default=None,
        help='[parameter] patch size [sz X sz]')
    parser.add_argument(
        '-c', '--crop_size', type=int, default=None,
        help='[parameter] centre crop size [cs X cs]')
    parser.add_argument(
        '--backbone', type=str, default='none',
        choices=[
            'none', 'default', 'resnet18', 'alexnet', 'inception_v3', 'mobilenet_v2'],
        help='choice of backbone neural network')
    parser.add_argument(
        '--cpu', action='store_true',
        help='force use of cpu')
    parser.add_argument(
        '--debug', action='store_true',
        help='log state debug')
    parser.add_argument(
        '--verbose', action='store_true',
        help='log state verbose')
    parser.add_argument(
        '--quiet', action='store_true',
        help='log state quiet')

    return parser


def validate_args(args):
    ''' Validate input arguments and set defaults '''
    global LOG_STATE
    if args.quiet:
        LOG_STATE = _LOG_STATES['QUIET']
    if args.verbose:
        LOG_STATE = _LOG_STATES['VERBOSE']
    if args.debug:
        LOG_STATE = _LOG_STATES['DEBUG']

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(f'cannot find directory at {args.base_dir}')

    out_path, out_name = os.path.split(args.out_path)
    _, out_ext = os.path.splitext(out_name)

    if out_ext not in ['.json', '.csv']:
        raise ValueError(f'Output file must be csv or json, not {out_ext}')
    elif out_ext == '.json':
        args.as_csv = False
    else:
        args.as_csv = True

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    if args.weights is None or not os.path.isfile(args.weights):
        raise ValueError(
            f'No valid weights provided. Could not find weights at {args.weights}')

    if (args.nb_patches is None or
            args.patch_size is None or
                args.crop_size is None or
                    args.backbone == 'none'):
        args.fix_hyperparams = False

    if not args.fix_hyperparams:
        p, s, c, backbone = _infer_hyperparams(args.weights)
        args.nb_patches = p
        args.patch_size = s
        args.crop_size = c
        args.backbone = backbone


    return args


def _infer_hyperparams(name):
    ''' infers hyperparameters from weight name '''
    _, name = os.path.split(name)  # remove any paths before filename
    name, _ = os.path.splitext(name)  # remove file extension

    p = re.findall('np(\d+)', name)
    if len(p) > 1:
        _log('STATUS', 'warning: too many options for nb patches, using first')
    p = int(p[0])
    _log('VERBOSE', f'inferred number of patches = {p}')

    s = re.findall('psz(\d+)', name)
    if len(s) > 1:
        _log('STATUS', 'warning: too many options for patch size, using first')
    s = int(s[0])
    _log('VERBOSE', f'inferred patch size = {s}')

    c = re.findall('csz(\d+)', name)
    if len(c) > 1:
        _log('STATUS', 'warning: too many options for crop size, using first')
    c = int(c[0])
    _log('VERBOSE', f'inferred crop size = {c}')

    choices = [
        'default', 'resnet18', 'alexnet', 'inception_v3', 'mobilenet_v2']
    backbone = None
    for _backbone in choices:
        if _backbone in name:
            backbone = _backbone
            break
    if backbone is None:
        raise ValueError(f'Unable to infer backbone from {name}')
    else:
        _log('VERBOSE', f'inferred backbone = {backbone}')
    return p, s, c, backbone


if __name__ == '__main__':
    args = validate_args(generate_parser().parse_args())

    _log('DEBUG', str(args))

    main(args)
