import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import argparse
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

img_size = 28
patch_size = 4
in_channels = 1
embed_dim = 64
num_heads = 2
num_layers = 2
num_classes = 10
batch_size = 64
epochs = 30

num_patches = (img_size // patch_size) ** 2


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)  
        x = x.flatten(2).transpose(1, 2)
        return x


class MiniViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])


def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./.data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./.data', train=False, download=True, transform=transform)

    small_train = torch.utils.data.Subset(train_dataset, range(0, 3000))
    train_loader = DataLoader(small_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, test_dataset


def train(train_loader, model_path, device):
    model = MiniViT().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")


def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return correct / total


def load_model(model_path, device):
    model = MiniViT().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"警告: 模型文件 {model_path} 不存在，将使用随机初始化的权重")
    return model


def predict_and_show(model, test_dataset, device):
    idx = random.randint(0, len(test_dataset)-1)
    img, true_label = test_dataset[idx]
    
    model.eval()
    with torch.no_grad():
        img_input = img.unsqueeze(0).to(device)
        out = model(img_input)
        pred_label = torch.argmax(out, dim=1).item()
    
    plt.figure(figsize=(3,3))
    plt.imshow(img.squeeze(), cmap="gray")
    
    title = f"True: {true_label} | Pred: {pred_label}"
    plt.title(title, color="green" if true_label==pred_label else "red")
    
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train 或 test')
    parser.add_argument('--model_path', type=str, default='minivit_mnist.pth',
                        help='模型保存/加载路径')
    args = parser.parse_args()

    device = torch.device("cpu")
    train_loader, test_loader, test_dataset = get_data_loaders()

    if args.mode == 'train':
        train(train_loader, args.model_path, device)
        model = load_model(args.model_path, device)
        acc = eval_acc(model, test_loader, device)
        print(f"测试集准确率: {acc:.4f}")
        predict_and_show(model, test_dataset, device)
    else:
        model = load_model(args.model_path, device)
        acc = eval_acc(model, test_loader, device)
        print(f"测试集准确率: {acc:.4f}")
        predict_and_show(model, test_dataset, device)


if __name__ == '__main__':
    main()