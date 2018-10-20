suffix = ''

run_name = __name__.split('.')[-1] + suffix

backbone_specs = {
    'backbone_module': 'imagenet_models',
    'backbone_function': 'resnet34_backbone',
    'kwargs': {
        'pretrained': False,
        'channel_config': (1, 2, 2, 2),
        'channel_multiplier': 64,
    },
    'head_channel_multiplier': 64,
}

train_val_split_dir = 'train_val_split'

epochs_before_val = 4
