import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet34

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载训练集
train_data = datasets.CIFAR10(root='D:\\train', train=True,
                              download=True, transform=transform_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 下载并加载测试集
test_data = datasets.CIFAR10(root='D:\\test', train=False,
                             download=True, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# 自定义 ResNet34 模型
def ResNet34():
    model = resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ResNet34().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
num_epochs = 30
best_acc = 0.0

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

    # 更新学习率
    scheduler.step()

    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.3f}, Test Acc: {acc:.2f}%")

# 加载最佳模型进行预测示例
model = ResNet34().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted labels:", predicted)