import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# setting the random seed
torch.manual_seed(99)
torch.cuda.manual_seed(99)
torch.cuda.manual_seed_all(99)
np.random.seed(99)

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

# create the dataset class
class MBTIDataset(Dataset):
    def __init__(self, data, labels_list, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = self.data[labels_list].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text=self.data.posts[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
# create the data loaders function    
def getDataLoaders(data, labels_list, tokenizer, max_len, batch_size):
    # create the dataset
    dataset = MBTIDataset(data, labels_list, tokenizer, max_len)

    # split the data
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create the data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

    return train_data_loader, val_data_loader, test_data_loader