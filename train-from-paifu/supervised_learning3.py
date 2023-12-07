from torch.utils.data import Dataset
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.net(x.float())


class NumpyMemmapDataset(Dataset):
    def __init__(self, input_file, target_file):
        self.input_memmap = np.load(input_file, mmap_mode="r")
        self.target_memmap = np.load(target_file, mmap_mode="r")
        assert self.input_memmap.shape[0] == self.target_memmap.shape[0], "Inconsistent dataset sizes"

    def __len__(self):
        return self.input_memmap.shape[0]

    def __getitem__(self, idx):
        input_data = self.input_memmap[idx]
        target_data = self.target_memmap[idx]
        return torch.tensor(input_data, dtype=torch.float), torch.tensor(target_data, dtype=torch.long)


def learn():
    print("Setting up dataset and dataloader")

    dataset = NumpyMemmapDataset("./shanten_obs_full_full.npy", "./shanten_actions_full_full.npy")
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=15)

    print("Initializing model")
    model = MLP()
    trainer = pl.Trainer(max_epochs=10, accelerator="mps", devices="1")

    print("Starting training")
    trainer.fit(model=model, train_dataloaders=loader)

    print("Training complete, saving model")
    torch.save(model.state_dict(), "./model_paifu_1_batch_size1024_10epochs_full_full.pth")
    print("Model saved")


learn()
