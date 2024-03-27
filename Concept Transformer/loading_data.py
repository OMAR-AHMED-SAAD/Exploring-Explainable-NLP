import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
from sklearn.model_selection import train_test_split


class EnterpreneurDataset(Dataset):
    def __init__(self, text,targets,attributes, tokenizer, max_len,num_classes=5):
        self.data = text
        self.targets = targets
        self.attributes = attributes
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        data = str(self.data[item])
        target = self.targets[item]
        attribute = self.attributes[item]
        encoding = self.tokenizer.encode_plus(
            data,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'data_text': data,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long),
            'attributes': F.one_hot(torch.tensor(attribute), self.num_classes).float()
        }    

# might need to update this function to handle the data in the right way  
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = EnterpreneurDataset(
        text = df.text.to_numpy(),
        targets = df.target.to_numpy(),
        attributes = df.label_id.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def split_load_data(df, tokenizer, max_len, batch_size,test_size=0.1,val_size=0.1):
    traain_df, test_df = train_test_split(df, test_size=test_size+val_size, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=test_size/(test_size+val_size), random_state=42)
    print(f"Train data shape: {traain_df.shape}" , f"Val data shape: {val_df.shape}", f"Test data shape: {test_df.shape}")
    train_data_loader = create_data_loader(traain_df, tokenizer, max_len, batch_size)
    val_data_loader = create_data_loader(val_df, tokenizer, max_len, batch_size)
    test_data_loader = create_data_loader(test_df, tokenizer, max_len, batch_size)
    return train_data_loader, val_data_loader, test_data_loader