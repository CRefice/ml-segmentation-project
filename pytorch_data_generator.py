import torch


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor, self.target_tensor = input_tensor, target_tensor    
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_tensor[idx, :, :, :]
        mask = self.target_tensor[idx, :, :, :]
        
        return [image, mask]