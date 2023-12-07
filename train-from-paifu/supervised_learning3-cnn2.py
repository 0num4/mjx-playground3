from torch.utils.data import Dataset
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class ModifiedCNN(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(32 * (obs_size // 4), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.unsqueeze(1)  # データを1次元の畳み込み層に適した形状に変更
        x = self.conv_net(x)
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
        #  rnn_input_size でエラーを吐くので、とりあえずコメントアウト
        rnn_input_size = 4352 // seq_length
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.rnn = nn.LSTM(input_size=32 * (obs_size // 4), hidden_size=hidden_size, num_layers=1, batch_first=True)
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
        # 畳み込み層の出力サイズを確認
        print("Conv net output shape:", x.shape)
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

    dataset = NumpyMemmapDataset("./shanten_obs_full.npy", "./shanten_actions_full.npy")
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=15)

    print("Initializing model")
    model = ModifiedCNN()
    trainer = pl.Trainer(max_epochs=10, accelerator="cuda", devices="1")

    print("Starting training")
    trainer.fit(model=model, train_dataloaders=loader)

    print("Training complete, saving model")
    torch.save(model.state_dict(), "./model_paifu_1_batch_size1024_10epochs_full_enhanced_cnn.pth")
    print("Model saved")


learn()
