suffix = '_1x1'

run_name = __name__.split('.')[-1] + suffix

backbone_specs = {
    'backbone_module': 'custom_models',
    'backbone_function': 'simple_backbone',
    'kwargs': {},
    'head_channel_multiplier': 64,
}

train_val_split_dir = 'train_val_split'

epochs_before_val = 4
