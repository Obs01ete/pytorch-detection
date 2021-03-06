suffix = ''

run_name = __name__.split('.')[-1] + suffix

backbone_specs = {
    'backbone_module': 'imagenet_models',
    'backbone_function': 'resnet50_backbone',
    'kwargs': {
        'pretrained': True,
    },
    'head_channel_multiplier': 128,
}

multibox_specs = {
    'use_ohem': True
}

train_val_split_dir = 'train_val_split'

epochs_before_val = 4
