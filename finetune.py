import warnings
warnings.filterwarnings("ignore")

from momentfm.utils.utils import control_randomness
from momentfm import MOMENTPipeline
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import os

control_randomness(seed=42)

class ReflectanceDataset(Dataset):
    # ... (This class remains unchanged) ...
    def __init__(self, dataframe, seq_len, data_stride_len):
        self.seq_len = seq_len
        self.data_stride_len = data_stride_len
        self.spec_cols = [c for c in dataframe.columns if c != "timestamp"]
        self.data = dataframe[self.spec_cols].values.astype('float32')
        self.length_timeseries = self.data.shape[0]
        self._generate_sequences()

    def _generate_sequences(self):
        self.sequences = []
        for idx in range(0, self.length_timeseries - self.seq_len + 1, self.data_stride_len):
            window = self.data[idx : idx + self.seq_len, :]
            self.sequences.append(window.T)

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return len(self.sequences)

def finetune_reflectance(seq_len, num_epochs, input_dataframe, batch_size=16, mask_rate=0.15):
    model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device).float()

    dataset = ReflectanceDataset(dataframe=input_dataframe, seq_len=seq_len, data_stride_len=seq_len // 2)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # --- ADDITION 1: Create the validation DataLoader ---
    # `shuffle=False` is important for validation to have consistent results.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 

    print(f"Starting fine-tuning... Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        # --- Training Loop (unchanged) ---
        model.train()
        train_losses = []
        for batch_x in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            batch_x = batch_x.reshape(-1, 1, seq_len).to(device)
            input_mask = (~torch.isnan(batch_x)).long()
            batch_x = torch.nan_to_num(batch_x, nan=0.0)
            pretrain_mask = (torch.rand(batch_x.shape, device=device) < (1 - mask_rate)).long()
            pretrain_mask = pretrain_mask.squeeze(1)
            
            output = model(x_enc=batch_x, input_mask=input_mask.squeeze(1), mask=pretrain_mask)
            loss = criterion(output.reconstruction[input_mask.bool()], batch_x[input_mask.bool()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # --- ADDITION 2: Validation Loop ---
        model.eval()  # Set the model to evaluation mode
        val_losses = []
        
        # `torch.no_grad()` disables gradient calculation to save memory and compute
        with torch.no_grad():
            for batch_x in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
                # The logic is identical to the training loop, just without backpropagation
                batch_x = batch_x.reshape(-1, 1, seq_len).to(device)
                input_mask = (~torch.isnan(batch_x)).long()
                batch_x = torch.nan_to_num(batch_x, nan=0.0)
                
                # Use a fixed mask for validation for consistent results (optional but good practice)
                # Here we will just use the same random masking for simplicity
                pretrain_mask = (torch.rand(batch_x.shape, device=device) < (1 - mask_rate)).long()
                pretrain_mask = pretrain_mask.squeeze(1)

                output = model(x_enc=batch_x, input_mask=input_mask.squeeze(1), mask=pretrain_mask)
                val_loss = criterion(output.reconstruction[input_mask.bool()], batch_x[input_mask.bool()])
                
                val_losses.append(val_loss.item())

        # --- ADDITION 3: Print both training and validation loss ---
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    return model