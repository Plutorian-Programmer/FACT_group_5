import argparse

def arg_parser_preprocessing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentires_dir", dest="sentires_dir", type=str, default="data/other_group/reviews_with_features.txt", 
                        help="path to sentires data")
    parser.add_argument("--review_dir", dest="review_dir", type=str, default="data/other_group/reviews_Electronics_5_filtered.json", 
                        help="path to original review data")
    parser.add_argument("--user_thresh", dest="user_thresh", type=int, default=20, 
                        help="remove users with reviews less than this threshold")
    parser.add_argument("--item_thresh", dest="item_thresh", type=int, default=20, 
                        help="remove users with reviews less than this threshold")
    parser.add_argument("--sample_ratio", dest="sample_ratio", type=int, default=2, 
                        help="the (negative: positive sample) ratio for training BPR loss")
    parser.add_argument("--test_length", dest="test_length", type=int, default=5, 
                        help="the number of test items")
    parser.add_argument("--val_length", dest="val_length", type=int, default=1, 
                        help="the number of val items")
    parser.add_argument("--neg_length", dest="neg_length", type=int, default=100, help="# of negative samples in evaluation")
    parser.add_argument("--save_path", dest="save_path", type=str, default="models/Dataset_20.pickle", 
                        help="The path to save the preprocessed dataset object")
    parser.add_argument("--use_pre", dest="use_pre", type=str, default=False, 
            help="The path to save the preprocessed dataset object")
    return parser.parse_args()



def arg_parser_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", dest = "device", type=str, default='cpu')
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--lr", dest="lr", type=float, default=0.01)
    parser.add_argument("--rec_k", dest="rec_k", type=int, default=5, help="length of rec list")
    parser.add_argument("--gpu", default=False)
    parser.add_argument("--weight_decay", default=0., type=float) # not sure whether to use
    parser.add_argument("--output_path", dest="output_path", type=str, default="models/model_20.model", 
                        help="The path to save the model")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    return parser.parse_args()

def arg_parser_CEF():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", dest = "device", type=str, default='cpu')
    parser.add_argument("--lr", dest="lr", type=float, default=0.01)
    parser.add_argument("--epochs", dest="epochs", type=int, default=3)
    parser.add_argument("--lambda", dest="ld", type=float, default=1.0)
    parser.add_argument("--beta", dest="beta", type=float, default=0.1)
    return parser.parse_args()
    
