'''
    Author: CentraleSupelec
    Year: 2021
    Python Version: >= 3.7
'''

import argparse
from torch.utils.data import DataLoader

from .dataset import PredictionsDataset
from .utils import create_session, get_label_threshold, collate_fn
from .models import Conflation, SanityCheck, TransformerAggregator, SanityCheckTransformer
from .training import train_and_validate
from .testing import test

def main(args):
    """
    Trains a model that predicts survival times using multiple medical reports
    """
    path_dataset, _, device, config = create_session(args)

    assert (768 + args.time_dim) % args.nhead == 0, f'd_model (i.e. 768 + time_dim) must be divisible by nhead. Found time_dim {args.time_dim} and nhead {args.nhead}'

    config.label_threshold = get_label_threshold(config, path_dataset)

    if not args.aggregator in ['conflation', 'sanity_check']:
        train_dataset, test_dataset = PredictionsDataset.get_train_validation(
            path_dataset, config, output_hidden_states=True, device=device)
    else:
        test_dataset = PredictionsDataset.get_train_validation(path_dataset, config, device=device, get_only_val=True)

    if not args.aggregator in ['conflation', 'sanity_check']:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    if args.aggregator == 'transformer':
        model = TransformerAggregator(device, config, args.nhead, args.num_layers, args.out_dim, args.time_dim)
        train_and_validate(model, train_loader, test_loader, device, config, config.path_result)  

    elif args.aggregator == 'sanity_check_transformer':
        model = SanityCheckTransformer(device, config)
        train_and_validate(model, train_loader, test_loader, device, config, config.path_result)    

    elif args.aggregator == 'conflation':
        model = Conflation(device, config)
        test(model, test_loader, config, config.path_result)

    elif args.aggregator == 'sanity_check':
        model = SanityCheck(device, config)
        test(model, test_loader, config, config.path_result)
        
    else:
        raise ValueError(f"Invalid aggregator name. Found {args.aggregator}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-a", "--aggregator", type=str, default="gru", 
        help="aggregator name", choices=['conflation', 'sanity_check', 'sanity_check_transformer', 'transformer'])
    parser.add_argument("-r", "--resume", type=str, required=True, 
        help="result folder in which the saved checkpoint will be reused")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-nr", "--nrows", type=int, default=None, 
        help="maximum number of samples for training and validation")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
        help="prints training loss every k batch")
    parser.add_argument("-dt", "--days_threshold", type=int, default=365, 
        help="days threshold to convert into classification task")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
        help="model learning rate")
    parser.add_argument("-wg", "--weight_decay", type=float, default=0, 
        help="the weight decay for L2 regularization")
    parser.add_argument("-p", "--patience", type=int, default=4, 
        help="number of decreasing accuracy epochs to stop the training")
    parser.add_argument("-me", "--max_ehrs", type=int, default=4, 
        help="maximum number of ehrs to be used for multi ehrs prediction")
    parser.add_argument("-nh", "--nhead", type=int, default=4, 
        help="number of transformer heads")
    parser.add_argument("-nl", "--num_layers", type=int, default=2, 
        help="number of transformer layers")
    parser.add_argument("-od", "--out_dim", type=int, default=2, 
        help="trasnformer out_dim (1 regression or 2 density)")
    parser.add_argument("-td", "--time_dim", type=int, default=8, 
        help="trasnformer time_dim")

    main(parser.parse_args())