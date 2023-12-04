import torch
from torch import optim, nn, utils, Tensor
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


def learn():
    import numpy as np
    inps = np.load("./shanten_obs_full.npy")
    tgts = np.load("./shanten_actions_full.npy")

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
    loader = DataLoader(dataset, batch_size=1024)

    model = MLP()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model=model, train_dataloaders=loader)
    torch.save(model.state_dict(), './model_paifu_1_bactch_size1024_10epochs.pth')


learn()
