class Config:
    voc_path = None
    model_name = "camembert-base"
    classify = False
    print_every_k_batch = 8
    max_size = 100000
    batch_size = 64
    learning_rate = 1e-5
    epochs = 20
    freeze = False
    weight_decay = 0
    ratio_lr_embeddings = None
    drop_rate = None
    train_size = 0.8

    def __init__(self, args):
        for attr in dir(self):
            if not attr.startswith('__') and hasattr(args, attr):
                setattr(self, attr, getattr(args, attr))