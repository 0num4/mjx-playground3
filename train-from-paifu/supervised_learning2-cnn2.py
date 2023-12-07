import numpy as np
import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


class EnhancedCNN(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=128, seq_length=10):
        super().__init__()
        self.seq_length = seq_length
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.rnn = nn.LSTM(input_size=32 * (obs_size // 4), hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc_net = nn.Sequential(
            nn.Linear(hidden_size * seq_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.unsqueeze(1)  # データを1次元の畳み込み層に適した形状に変更
        batch_size = x.size(0)
        x = self.conv_net(x)
        x = x.view(batch_size, self.seq_length, -1)
        x, _ = self.rnn(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.fc_net(x)
        return x

    def training_step(self, batch, batch_idx):
        # トレーニングステップのロジックをここに記述
        x, y = batch
        logits = self(x)
        loss = self.loss_module(logits, y)
        return loss

    def train_dataloader(self):
        # トレーニングデータローダーの定義
        inps = np.load("./shanten_obs_full_full.npy", mmap_mode="r")
        tgts = np.load("./shanten_actions_full_full.npy", mmap_mode="r")
        dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
        return DataLoader(dataset, batch_size=1024, num_workers=15)

    def configure_optimizers(self):
        # オプティマイザーの設定
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def learn():
    model = EnhancedCNN()
    trainer = pl.Trainer(max_epochs=10, accelerator="cuda", devices="1")
    print("training start")
    trainer.fit(model=model)
    print("training end")
    torch.save(model.state_dict(), './model_paifu_1_batch_size1024_10epochs_noCustomDataloader_full_full_cnn_enhanced.pth')
    print("saved model")


learn()
