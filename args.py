import argparse

def arg_parser_preprocessing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentires_dir", dest="sentires_dir", type=str, default="data/reviews_small.txt", 
                        help="path to sentires data")
    parser.add_argument("--review_dir", dest="review_dir", type=str, default="data/Electronics.json", 
                        help="path to original review data")
    parser.add_argument("--user_thresh", dest="user_thresh", type=int, default=2, 
                        help="remove users with reviews less than this threshold")
    parser.add_argument("--item_thresh", dest="item_thresh", type=int, default=2, 
                        help="remove users with reviews less than this threshold")
    parser.add_argument("--sample_ratio", dest="sample_ratio", type=int, default=2, 
                        help="the (negative: positive sample) ratio for training BPR loss")
    parser.add_argument("--test_length", dest="test_length", type=int, default=1, 
                        help="the number of test items")
    parser.add_argument("--val_length", dest="val_length", type=int, default=0, 
                        help="the number of val items")
    parser.add_argument("--neg_length", dest="neg_length", type=int, default=100, help="# of negative samples in evaluation")
    return parser.parse_args()



def arg_parser_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", dest = "device", type=str, default='cpu')
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argement("--lr", dest="lr", type=float, default=0.01)
    parser.add_argument("--rec_k", dest="rec_k", type=int, default=5, help="length of rec list")

