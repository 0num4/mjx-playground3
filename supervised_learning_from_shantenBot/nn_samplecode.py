from torch import optim, nn  # utils, Tensor
import pytorch_lightning as pl

"""
このMLPとは、"Multi-Layer Perceptron"（多層パーセプトロン）の略です。Multi-Layer Perceptronは、深層学習モデルの一つで、複数の隠れ層（hidden layer）を持つ人工ニューラルネットワークの一種です。このモデルは、入力層、複数の隠れ層、そして出力層から構成されており、非線形関数（通常はReLUなど）を用いて層間の情報伝播を行います。
"""

#  クラスを作る
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