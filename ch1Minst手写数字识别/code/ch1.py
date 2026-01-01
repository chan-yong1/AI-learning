import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------
# 1) 超参数
# ----------------------------
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "mnist_cnn.pth"

# ----------------------------
# 2) 数据预处理
# ToTensor: [0,255] -> [0,1]
# Normalize: 标准化（更容易收敛）
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST常用均值方差
])

# ----------------------------
# 3) 数据集
# ----------------------------
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Windows 若 DataLoader 卡住/报错，把 num_workers 改为 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ----------------------------
# 4) 模型：简单 CNN
# ----------------------------
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # (N,32,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # (N,32,14,14)

            nn.Conv2d(32, 64, 3, padding=1), # (N,64,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # (N,64,7,7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MNIST_CNN().to(DEVICE)

# ----------------------------
# 5) 损失与优化器
# CrossEntropyLoss = LogSoftmax + NLLLoss（所以模型最后不要手动 softmax）
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# 6) 评估函数
# ----------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

# ----------------------------
# 7) 训练循环
# ----------------------------
def train():
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(test_loader)

        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> saved best model: {SAVE_PATH} (best_acc={best_acc:.4f})")

    print("Training finished.")

if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    train()
