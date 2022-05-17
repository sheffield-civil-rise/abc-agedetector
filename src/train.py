import argparse
import os
import time
import copy

import tqdm  # progress bar

import torch
import torchvision
from torch.utils.data import DataLoader

from utils import is_valid_file
from dataset import generate_dataset, get_train_test_samplers
from transforms import DefaultTransformation
from model import AgeDetector, BaseConvNet


_LOG_STATES = {'DEBUG': 0, 'VERBOSE': 1, 'STATUS': 2, 'QUIET': 3}

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


def _setup_parallel_enviromment(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(
        'nccl', rank=rank, world_size=world_size)


def setup_parallel_environment():
    # i don't know what i'm doing
    NotImplemented


def prepare_dataset_loader(args):
    ''' '''
    _log('VERBOSE', 'Generating dataset folders')
    transforms = DefaultTransformation(
        args.nb_patches, args.patch_size, args.crop_size)

    imagefolders = [generate_dataset(
        args.base_dir,
        is_valid_file=is_valid_file,
        transforms=transforms) for _ in range(2)]

    _log('VERBOSE', 'Creating samplers')
    nb_data = len(imagefolders[0])
    samplers = get_train_test_samplers(
        nb_data,
        ratio=(80, 20),
        seed=args.seed)

    _log('VERBOSE', 'Creating data loaders')
    train_loader, test_loader = [DataLoader(
        imagefolder,
        batch_size=args.batch_size,
        sampler=sampler)
        for imagefolder, sampler in zip(imagefolders, samplers)]

    return train_loader, test_loader


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


def create_model(args, nb_classes):
    ''' create model and load any weights if required '''
    backbone = get_backbone(args)

    _log('VERBOSE', f'Creating model with {nb_classes} classes')
    model = AgeDetector(
        nb_classes,
        backbone=backbone,
        pretrained=args.pretrained_backbone,
        fixbackbone=args.fix_backbone)

    if args.pretrained:
        _log('VERBOSE', f'Loading pretrained weights from {args.weights}')
        state_dict = torch.load(args.weights)
        model.load_state_dict(state_dict)

    model.eval()

    return model


def save_weights(model, args):
    ''' writes out model weights to out_path '''
    _log('VERBOSE', 'Extracting state dictionary')
    model.eval()
    state_dict = model.state_dict()
    _log('STATUS', f'Saving weights at {args.out_path}')
    torch.save(state_dict, args.out_path)


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    num_epochs=25):
    ''' training sequence '''

    since = time.time()

    _log('VERBOSE', f'Moving model to device:{device}')
    # load model to device
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = [
        len(dataloader.sampler.indices)
        for dataloader in dataloaders]

    _log('VERBOSE', f'Training on {dataset_sizes[0]} images')
    _log('VERBOSE', f'Validating on {dataset_sizes[1]} images')

    for epoch in range(num_epochs):
        _log('STATUS', f'Epoch {epoch}/{num_epochs - 1}')
        _log('STATUS', '-'*10)

        ##
        for idx, phase in enumerate(['train', 'val']):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            # iterate over data
            with tqdm.tqdm(dataloaders[idx], unit='batch') as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(['train', '  val'][idx])

                    inputs = inputs.to(device)  # move to gpu/cpu
                    labels = labels.to(device)  # move to gpu/cpu

                    # reset optimizer gradients
                    optimizer.zero_grad()

                    # forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)           # class probabilities
                        _, preds = torch.max(outputs, 1)  # class guess
                        loss = criterion(outputs, labels)

                        # backpass / optimize for train
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # get stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    # print stats
                    _ploss = running_loss/dataset_sizes[idx]
                    _paccu = 100*running_corrects.item()/dataset_sizes[idx]
                    tepoch.set_postfix(loss=_ploss, accuracy=_paccu)
                    time.sleep(0.1)
            if phase == 'train' and scheduler is not None:
                scheduler.step()  # update parameter tuning scheduler

            epoch_loss = running_loss/dataset_sizes[idx]
            epoch_acc = running_corrects.double()/dataset_sizes[idx]

            _log('STATUS', f'{phase}, Loss:{epoch_loss:.4f} Acc:{epoch_acc}')

            # deep copy best model
            if phase == 'val' and epoch_acc > best_acc:
                _log(
                    'VERBOSE',
                    f'Updating best weights (acc={epoch_acc}<{best_acc})')
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        _log('STATUS', '')
    elapsed = time.time() - since

    _log(
        'STATUS',
        f'Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    _log('STATUS', f'Best val acc: {best_acc:4f}')

    # return best model
    model.load_state_dict(best_model_wts)
    return model


def main(args):
    ''' main body '''

    if args.seed is not None:
        _log('STATUS', f'Setting seed as {args.seed}')
        torch.manual_seed(args.seed)

    if args.parallel:
        _parallel_able = (
            torch.cuda.is_available() and torch.cuda.device_count() > 1)
        if not _parallel_able:
            _log(
                'STATUS',
                'Unable to parallelise because nb devices less than 2')
    else:
        _parallel_able = False

    if _parallel_able:
        setup_parallel_environment()

    train_loader, test_loader = prepare_dataset_loader(args)

    nb_classes = len(train_loader.dataset.classes)

    model = create_model(args, nb_classes)

    # generate device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    _log('STATUS', f'Using device {device}')
    _log('DEBUG', 'Is this right ?')

    if _parallel_able:
        _log('STATUS', 'Creating parallel model')
        model = torch.nn.parallel.DistributedDataParallel(model)

    # count parameters
    num_parameter = sum([p.numel() for p in model.parameters()])
    num_trainable = sum(
        [p.numel() for p in model.parameters() if p.requires_grad])
    _log(
        'VERBOSE',
        f'There are {num_parameter} parameters, of which {num_trainable} are trainable')

    # train code
    # ...
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = None
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if not args.notrain:
        model = train_model(
            model, criterion,
            optimizer, scheduler,
            (train_loader, test_loader),
            device,
            num_epochs=args.nb_epochs)

        # save weights
        save_weights(model, args)
    else:
        _log('DEBUG', 'Stopping because I was asked not to train the model')


def generate_parser():
    ''' Generates argument parser for commandline use '''
    # create parser object
    parser = argparse.ArgumentParser(
        description='Trains an age detection model')
    # add arguments
    parser.add_argument(
        'base_dir', type=str,
        help='directory containing images to train/test and class.json')
    parser.add_argument(
        'out_path', type=str,
        help='path to save model weights to')
    parser.add_argument(
        '-e', '--nb_epochs', type=int, default=25,
        help='number of epochs for training')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=1,
        help='minibatch size for training')
    parser.add_argument(
        '-p', '--nb_patches', type=int, default=9,
        help='[parameter] number of patches to extract from images')
    parser.add_argument(
        '-s', '--patch_size', type=int, default=128,
        help='[parameter] patch size [sz X sz]')
    parser.add_argument(
        '-c', '--crop_size', type=int, default=1024,
        help='[parameter] centre crop size [cs X cs]')
    parser.add_argument(
        '--seed', default=None,
        help='random seed (for repeatability)')
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to weights for using pretrained network')
    parser.add_argument(
        '--backbone', type=str, default='resnet18',
        choices=[
            'default', 'resnet18', 'alexnet', 'inception_v3', 'mobilenet_v2'],
        help='choice of backbone neural network')
    parser.add_argument(
        '--pretrained_backbone', action='store_true',
        help='use pretrained weights for backbone')
    parser.add_argument(
        '--fix_backbone', action='store_true',
        help='fix weights in backbone (will assume pretrained backbone even if not specified)')
    parser.add_argument(
        '--cpu', action='store_true',
        help='force use of cpu')
    parser.add_argument(
        '--parallel', action='store_true',
        help='use parallelisation on multiple gpu')
    parser.add_argument(
        '--debug', action='store_true',
        help='log state debug')
    parser.add_argument(
        '--verbose', action='store_true',
        help='log state verbose')
    parser.add_argument(
        '--notrain', action='store_true',
        help='prevent model training (for debugging)')
    parser.add_argument(
        '--quiet', action='store_true',
        help='log state quiet')

    return parser


def validate_args(args):
    ''' Validate input arguments and set defaults '''

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(f'cannot find directory at {args.base_dir}')
    if args.weights is not None:
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(
                f'cannot find model weights at {args.weights}')
        args.pretrained = True
    else:
        args.pretrained = False

    global LOG_STATE
    if args.quiet:
        LOG_STATE = _LOG_STATES['QUIET']
    if args.verbose:
        LOG_STATE = _LOG_STATES['VERBOSE']
    if args.debug:
        LOG_STATE = _LOG_STATES['DEBUG']

    return args


if __name__ == '__main__':
    args = validate_args(generate_parser().parse_args())

    _log('DEBUG', str(args))

    main(args)
