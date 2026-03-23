import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

# ========== 1. 加载数据 ==========
print("加载数据...")
text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_reg_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_reg_avg.npz'))['arr_0']

# 对每个样本的 3 个句子取平均
text_features = text_features.mean(axis=1)  # (162, 3, 1024) -> (162, 1024)

# 将回归标签转换为二分类（SDS >= 53 为抑郁）
text_targets = (text_targets >= 53).astype(int)

print(f"总样本数: {text_features.shape[0]}")
print(f"特征维度: {text_features.shape[1]}")
print(f"正样本数: {np.sum(text_targets == 1)} (抑郁症)")
print(f"负样本数: {np.sum(text_targets == 0)} (非抑郁)")
print("=" * 60)


# ========== 2. 定义 MLP 分类器（简化版） ==========
class TextMLP(nn.Module):
    def __init__(self, config):
        super(TextMLP, self).__init__()
        self.fc1 = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.dropout1 = nn.Dropout(config['dropout'])
        self.fc2 = nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2)
        self.dropout2 = nn.Dropout(config['dropout'])
        self.fc3 = nn.Linear(config['hidden_dim'] // 2, config['num_classes'])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


# ========== 3. 过采样函数 ==========
def oversample_features(features, targets, train_idx):
    """对训练集中的正样本进行过采样"""
    train_features = features[train_idx]
    train_targets = targets[train_idx]

    pos_idx = np.where(train_targets == 1)[0]
    neg_idx = np.where(train_targets == 0)[0]

    if len(pos_idx) > 0:
        # 复制正样本 3 次
        oversample_factor = 3
        oversampled_pos_idx = np.repeat(pos_idx, oversample_factor)
        final_idx = np.concatenate([oversampled_pos_idx, neg_idx])
        return train_features[final_idx], train_targets[final_idx]
    else:
        return train_features, train_targets


# ========== 4. 训练函数 ==========
def train_one_epoch(model, optimizer, criterion, X_train, y_train, config):
    model.train()
    total_loss = 0
    correct = 0

    # 打乱数据
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    for i in range(0, len(X_train), config['batch_size']):
        batch_X = X_train[i:i + config['batch_size']]
        batch_y = y_train[i:i + config['batch_size']]

        if config['cuda']:
            x = Variable(torch.from_numpy(batch_X).type(torch.FloatTensor)).cuda()
            y = Variable(torch.from_numpy(batch_y)).cuda()
        else:
            x = Variable(torch.from_numpy(batch_X).type(torch.FloatTensor))
            y = Variable(torch.from_numpy(batch_y))

        optimizer.zero_grad()
        output = model(x)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        y = y.long()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    accuracy = 100. * correct / len(X_train)
    return total_loss, accuracy


# ========== 5. 评估函数 ==========
def evaluate(model, X_test, y_test, config):
    model.eval()
    with torch.no_grad():
        if config['cuda']:
            x = Variable(torch.from_numpy(X_test).type(torch.FloatTensor)).cuda()
            y = Variable(torch.from_numpy(y_test)).cuda()
        else:
            x = Variable(torch.from_numpy(X_test).type(torch.FloatTensor))
            y = Variable(torch.from_numpy(y_test)).type(torch.LongTensor)

        output = model(x)
        pred = output.data.max(1, keepdim=True)[1]

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, pred.cpu().numpy())

        # 计算指标
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            accuracy = 0
            precision = recall = f1 = 0

        return accuracy, precision, recall, f1, cm


# ========== 6. 主程序：5折交叉验证 ==========
config = {
    'num_classes': 2,
    'input_dim': 1024,
    'hidden_dim': 256,
    'dropout': 0.5,
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 1e-3,
    'cuda': False,
}

# 设置类别权重（正样本权重更高）
class_weights = torch.FloatTensor([1.0, 3.0])  # 抑郁症权重3倍
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_f1_scores = []
all_recalls = []
all_precisions = []
all_accuracies = []

print("开始 5 折交叉验证...")
print("=" * 60)

for fold, (train_idx, test_idx) in enumerate(skf.split(text_features, text_targets)):
    print(f"\n========== Fold {fold + 1}/5 ==========")

    # 过采样训练集
    X_train, y_train = oversample_features(text_features, text_targets, train_idx)
    X_test = text_features[test_idx]
    y_test = text_targets[test_idx]

    print(f"训练集: {len(X_train)} 样本 (正样本: {np.sum(y_train == 1)})")
    print(f"测试集: {len(X_test)} 样本 (正样本: {np.sum(y_test == 1)})")

    # 初始化模型和优化器
    model = TextMLP(config)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    best_f1 = 0
    best_epoch = 0
    best_recall = 0

    for epoch in range(config['epochs']):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion,
                                                X_train, y_train, config)
        val_acc, val_prec, val_rec, val_f1, cm = evaluate(model, X_test, y_test, config)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:2d} | Loss: {train_loss:.2f} | Train Acc: {train_acc:.1f}% | "
                  f"Val Acc: {val_acc:.3f} | Prec: {val_prec:.3f} | Rec: {val_rec:.3f} | F1: {val_f1:.3f}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_recall = val_rec
            best_epoch = epoch + 1
            os.makedirs(os.path.join(prefix, 'Model/ClassificationWhole/Text'), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(prefix, f'Model/ClassificationWhole/Text/MLP_fold{fold + 1}_best.pt'))

    print(f"Fold {fold + 1} 最佳结果: F1={best_f1:.4f}, Recall={best_recall:.4f} (Epoch {best_epoch})")
    all_f1_scores.append(best_f1)
    all_recalls.append(best_recall)

    # 最后一次评估，显示混淆矩阵
    model.load_state_dict(
        torch.load(os.path.join(prefix, f'Model/ClassificationWhole/Text/MLP_fold{fold + 1}_best.pt')))
    final_acc, final_prec, final_rec, final_f1, cm = evaluate(model, X_test, y_test, config)
    print(f"最终测试结果:")
    print(f"  混淆矩阵: \n{cm}")
    print(f"  Accuracy: {final_acc:.4f}, Precision: {final_prec:.4f}, Recall: {final_rec:.4f}, F1: {final_f1:.4f}")

# 输出总结
print("\n" + "=" * 60)
print("5 折交叉验证结果汇总:")
print(f"F1 分数: {[f'{x:.4f}' for x in all_f1_scores]}")
print(f"Recall: {[f'{x:.4f}' for x in all_recalls]}")
print(f"平均 F1: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
print(f"平均 Recall: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
print("训练完成！")