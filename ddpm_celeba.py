from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # Dataset configurations
    config.dataset = 'celeba'
    config.image_size = 64
    config.num_channels = 3

    # Model configurations
    config.model = config_dict.ConfigDict()
    config.model.name = 'ddpm'
    config.model.channels = 128
    config.model.num_res_blocks = 2
    config.model.attention_resolutions = [16, 8]
    config.model.dropout = 0.1

    # Training configurations
    config.train = config_dict.ConfigDict()
    config.train.batch_size = 64
    config.train.num_steps = 700000
    config.train.lr = 0.0002

    # Evaluation configurations
    config.eval = config_dict.ConfigDict()
    config.eval.enable_sampling = True
    config.eval.enable_bpd = False
    config.eval.dataset = 'test'
    config.eval.num_samples = 10000

    return config
