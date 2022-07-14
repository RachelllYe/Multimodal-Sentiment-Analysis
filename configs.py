import argparse
from email.policy import default

from numpy import require
def add_args(parser):
    parser.add_argument("--work_dir", default='',type=str)
    parser.add_argument("--result_path", default='/result.txt',type=str)
    parser.add_argument("--test_file", default='/test_without_label.txt',type=str)
    parser.add_argument("--batch_size", default=16,type = int)
    parser.add_argument("--hidden_sz", default=768,type = int)
    parser.add_argument("--n_classes", default=3,type = int)
    parser.add_argument("--epochs", default=30,type = int)
    parser.add_argument("--num_image_embeds", default=3,type = int)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--model_type", type=str, default=True,choices=['multimodal', 'unimodal_img', 'unimodal_text'],required = True)
    
    args = parser.parse_args()
    return args

# parser = argparse.ArgumentParser(description="Train Models")
# add_args(parser)
# args, remaining_args = parser.parse_known_args()
# print(remaining_args)