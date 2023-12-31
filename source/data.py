import os, sys
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df_path, data_dir, 
    ):
        self.df_path, self.data_dir,  = df_path, data_dir, 
        self.df = pd.read_csv(self.df_path)

    def __len__(self, 
    ):
        return len(self.df)

    def __getitem__(self, 
        index, 
    ):
        instance = self.df.iloc[index]
        label0, label1 = instance["label0"], instance["label1"]

        image = cv2.imread("{}/{}".format(self.data_dir, instance["Id"]))
        image = cv2.cvtColor(
            image, 
            code = cv2.COLOR_BGR2GRAY, 
        )[..., np.newaxis]/255
        image = A.Compose(
            [
                AT.ToTensorV2(), 
            ]
        )(image = image)["image"]

        return image, [label0, label1]