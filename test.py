import torch
import time
import argparse

# print(time.time())
# print(torch.cuda.is_available())

parser = argparse.ArgumentParser()

parser.add_argument('--e',          type=int,   default=150)
parser.add_argument('--b',     type=int,   default=128)
parser.add_argument('--l',     type=float, default=0.1)
args = parser.parse_args()
print(args.b, args.e, args.l)
