import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models.models import MultiGateSMoE
from engines import train_fn

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
optim = optim.Adam(
    model.parameters(), weight_decay = 5e-5, 
    lr = 1e-3, 
)

save_ckp_dir = "../ckps/{}".format(args.dataset)
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, 20, 
    model, 
    optim, 
    save_ckp_dir, 
)