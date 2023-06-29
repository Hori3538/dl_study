import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from model import EfficientNet
from dataset import CifarDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet()
    model.to(device)

    train_dataset = CifarDataset("train")
    valid_dataset = CifarDataset("valid")

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    n_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 学習曲線表示用リスト
    loss_per_epoch_train = []
    acc_per_epoch_train = []
    loss_per_epoch_valid = []
    acc_per_epoch_valid = []

    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []

        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        model.train()  # 訓練時には勾配を計算するtrainモードにする
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, t)
            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)

            losses_train.append(loss.tolist())

            acc = torch.where(t.to("cpu") - pred.to("cpu") == 0, torch.ones_like(t.to("cpu")), torch.zeros_like(t.to("cpu")))
            train_num += acc.size()[0]
            train_true_num += acc.sum().item()

        model.eval()  # 評価時には勾配を計算しないevalモードにする
        for x, t in valid_loader:
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, t)

            _, pred = outputs.max(1)

            losses_valid.append(loss.tolist())

            acc = torch.where(t.to("cpu") - pred.to("cpu") == 0, torch.ones_like(t.to("cpu")), torch.zeros_like(t.to("cpu")))
            valid_num += acc.size()[0]
            valid_true_num += acc.sum().item()
            
        valid_acc = valid_true_num/valid_num
        loss_per_epoch_train.append(np.mean(losses_train))
        acc_per_epoch_train.append(train_true_num/train_num)
        loss_per_epoch_valid.append(np.mean(losses_valid))
        acc_per_epoch_valid.append(valid_acc)
        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch+1,
            np.mean(losses_train),
            train_true_num/train_num,
            np.mean(losses_valid),
            valid_acc
        ))

if __name__ == "__main__":
    main()

