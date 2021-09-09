from torch import nn

import resnet
from utils import _construct_depth_model


def generate_model(args):

    if args.pre_train_model == False or args.mode == 'test':
        print('Without Pre-trained model')
        assert args.model_depth in [18, 50, 101]
        if args.model_depth == 18:

            model = resnet.resnet18(
                output_dim=args.feature_dim,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration,
                shortcut_type=args.shortcut_type,
                tracking=args.tracking,
                pre_train=args.pre_train_model
            )

        elif args.model_depth == 50:
            model = resnet.resnet50(
                output_dim=args.feature_dim,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration,
                shortcut_type=args.shortcut_type,
                tracking=args.tracking,
                pre_train=args.pre_train_model
            )

        elif args.model_depth == 101:
            model = resnet.resnet101(
                output_dim=args.feature_dim,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration,
                shortcut_type=args.shortcut_type,
                tracking=args.tracking,
                pre_train=args.pre_train_model
            )

        model = nn.DataParallel(model, device_ids=None)

        model = _construct_depth_model(model)
    if args.use_cuda:
        model = model.cuda()
    return model
