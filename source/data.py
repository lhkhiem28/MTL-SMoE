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
        label1, label2 = instance["label1"], instance["label2"]

        image = cv2.imread("{}/{}".format(self.data_dir, instance["Id"]))
        image = cv2.cvtColor(
            image, 
            code = cv2.COLOR_BGR2GRAY, 
        )[..., np.newaxis]
        image = A.Compose(
            [
                A.Normalize(
                    mean = (0.1307), std = (0.3081), 
                ), 
                AT.ToTensorV2(), 
            ]
        )(image = image)["image"]

        return image, (label1, label2)