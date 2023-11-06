import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models.models import MultiGateSMoE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "Multi-MNIST")
args = parser.parse_args()

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            df_path = "../datasets/{}/train.csv".format(args.dataset), data_dir = "../datasets/{}/train".format(args.dataset), 
        ), 
        num_workers = 8, batch_size = 512, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            df_path = "../datasets/{}/val.csv".format(args.dataset), data_dir = "../datasets/{}/val".format(args.dataset), 
        ), 
        num_workers = 8, batch_size = 512, 
        shuffle = True, 
    ), 
}
model = MultiGateSMoE(
    num_classes = 10, 
)