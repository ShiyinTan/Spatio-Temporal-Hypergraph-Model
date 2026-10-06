import os
import random
import logging
import torch
import numpy as np
import os.path as osp


def get_root_dir():
    """Repository root, independent of the current working directory and folder name."""
    return osp.dirname(osp.dirname(osp.abspath(__file__)))


def torch_load(path, map_location=None):
    """Load a checkpoint or PyG data file.

    PyTorch 2.6 defaults ``weights_only=True``, which rejects the pickled
    ``Data`` objects and optimizer state this project stores.
    """
    kwargs = {'weights_only': False}
    if map_location is not None:
        kwargs['map_location'] = map_location
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop('weights_only')
        return torch.load(path, **kwargs)


def set_logger(args):
    """
    Write logs to checkpoint and console
    """
    if args.do_train:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'train.log')
    else:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'test.log')

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file, mode='w+')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
