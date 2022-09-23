'''
    Centre Léon Bérard, Théo Di Piazza (Data - Recherche)
    2022
    Script which permits to train kmembert-T2-classification

    Please, make sure to :
    - activate clb_env2
    - python fineTuning_DataTest.py | python3 fineTuning_DataTest.py
    - Have data in - data, ehr - repository
'''

# Import libraries
if(True):
    # From other scripts
    from kmembert.utils import Config
    from kmembert.models import TransformerAggregator
    from kmembert.utils import get_root, now
    from kmembert.dataset import PredictionsDataset
    from kmembert.utils import create_session, get_label_threshold, collate_fn, collate_fn_with_id
    from kmembert.training import train_and_validate

    # Classic
    import os
    import argparse
    import torch
    from torch.utils.data import DataLoader

# 1/4 - Load the Model
if(True):
    resume = "kmembert-T2"
    config = Config()
    config.resume = resume

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nhead, num_layers, out_dim, time_dim = 8, 4, 2, 8

    # Init model
    model = TransformerAggregator(device, config, nhead, num_layers, out_dim, time_dim)

    # Load the model
    model.resume(config)

# 2/4 - Import ArgParse Manually
if(True):
    # ArgParse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-a", "--aggregator", type=str, default="transformer", 
        help="aggregator name", choices=['conflation', 'sanity_check', 'sanity_check_transformer', 'transformer'])
    parser.add_argument("-r", "--resume", type=str, default = "kmembert-base", 
        help="result folder in which the saved checkpoint will be reused")
    parser.add_argument("-e", "--epochs", type=int, default=25, 
        help="number of epochs")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and validation")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
        help="prints training loss every k batch")
    parser.add_argument("-dt", "--days_threshold", type=int, default=365, 
        help="days threshold to convert into classification task")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, 
        help="model learning rate")
    parser.add_argument("-wg", "--weight_decay", type=float, default=0, 
        help="the weight decay for L2 regularization")
    parser.add_argument("-p", "--patience", type=int, default=4, 
        help="number of decreasing accuracy epochs to stop the training")
    parser.add_argument("-me", "--max_ehrs", type=int, default=4, 
        help="maximum nusmber of ehrs to be used for multi ehrs prediction")
    parser.add_argument("-nh", "--nhead", type=int, default=8, 
        help="number of transformer heads")
    parser.add_argument("-nl", "--num_layers", type=int, default=4, 
        help="number of transformer layers")
    parser.add_argument("-od", "--out_dim", type=int, default=2, 
        help="transformer out_dim (1 regression or 2 density)")
    parser.add_argument("-td", "--time_dim", type=int, default=8, 
        help="transformer time_dim")

    args = parser.parse_args("")

# 3/4 - Load Dataset and DataLoader
if(True):
    # Load dataset and dataloader
    path_dataset, _, device, config = create_session(args)

    assert (768 + args.time_dim) % args.nhead == 0, f'd_model (i.e. 768 + time_dim) must be divisible by nhead. Found time_dim {args.time_dim} and nhead {args.nhead}'

    config.label_threshold = get_label_threshold(config, path_dataset)

    train_dataset, test_dataset = PredictionsDataset.get_train_validation(
        path_dataset, config, output_hidden_states=True, device=device)

    if not args.aggregator in ['conflation', 'sanity_check']:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 4/4 - Train and Validate the Model
train_and_validate(model, train_loader, test_loader, device, config, config.path_result)  