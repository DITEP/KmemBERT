from torch.utils.data import Dataset
import pandas as pd

class TweetDataset(Dataset):
    def __init__(self, csv_name):
        super(TweetDataset, self).__init__()
        self.csv_name = csv_name
        self.df = pd.read_csv(self.csv_name)
        self.labels = list(self.df.label)
        self.texts = list(self.df.text)
        
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)