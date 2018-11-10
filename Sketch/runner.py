import os
import predict_views
import argparse

parser =argparse.ArgumentParser()
parser.add_argument("--input_dir","-i")
args=parser.parse_args()
#print(args)
model_dir=args.input_dir
#model_dir='/media/vishnu/New Volume/Holojest/slave/Datasets/00049_Grimmjow_Res'
predict_views.collect(model_dir)
