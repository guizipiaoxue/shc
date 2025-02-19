import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据增强和标准化
# 计算训练集的均值和标准差
train_dataset = datasets.CIFAR10(root='D:\\train', train=True, download=True)
train_data = train_dataset.data / 255.0
mean = train_data.mean(axis=(0, 1, 2))
std = train_data.std(axis=(0, 1, 2))

# 训练集数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 测试集数据处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 下载并加载训练集
train_data = datasets.CIFAR10(root='D:\\train', train=True,
                              download=True, transform=transform_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 下载并加载测试集
test_data = datasets.CIFAR10(root='D:\\test', train=False,
                             download=True, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# 改进的 LeNet 模型
class ImprovedLeNet(nn.Module):
    def __init__(self):
        super(ImprovedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ImprovedLeNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
num_epochs = 30
best_acc = 0.0
patience = 5
no_improvement_count = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print("Early stopping!")
            break

    # 更新学习率
    scheduler.step()

    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.3f}, Test Acc: {acc:.2f}%")


# 加载最佳模型进行预测示例
model = ImprovedLeNet().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted labels:", predicted)