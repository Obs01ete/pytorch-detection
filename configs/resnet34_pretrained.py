suffix = ''

run_name = __name__.split('.')[-1] + suffix

backbone_specs = {
    'backbone_module': 'imagenet_models',
    'backbone_function': 'resnet34_backbone',
    'kwargs': {
        'pretrained': True,
    },
    'head_channel_multiplier': 128,
}

train_val_split_dir = 'train_val_split'
