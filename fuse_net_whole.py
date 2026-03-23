import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import os
import pickle

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

# ========== 1. 加载数据 ==========
print("加载数据...")
text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_reg_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_reg_avg.npz'))['arr_0']

# 加载音频特征（pickle 格式，变长序列）
with open(os.path.join(prefix, 'Features/AudioWhole/whole_samples_reg_256.pkl'), 'rb') as f:
    audio_features = pickle.load(f)
with open(os.path.join(prefix, 'Features/AudioWhole/whole_labels_reg_256.pkl'), 'rb') as f:
    audio_targets = pickle.load(f)

# 将回归标签转换为二分类
text_targets = (text_targets >= 53).astype(int)
# audio_targets 是列表，需要先转成 numpy 数组
audio_targets = np.array(audio_targets)
audio_targets = (audio_targets >= 53).astype(int)

print(f"原始文本特征: {text_features.shape}")
print(f"原始音频特征数量: {len(audio_features)}")
print(f"正样本: {np.sum(text_targets == 1)}, 负样本: {np.sum(text_targets == 0)}")

# ========== 2. 划分数据集 ==========
X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
    text_features, audio_features, text_targets, test_size=0.3, random_state=42, stratify=text_targets
)

# ========== 3. 过采样（处理变长音频）==========
print("\n过采样前训练集正样本数:", np.sum(y_train == 1))

# 过采样只能用于固定维度的特征，音频需要特殊处理
# 先把文本特征展平
X_text_flat = X_text_train.reshape(len(X_text_train), -1)  # (N, 3*1024=3072)

# 音频特征无法直接过采样，我们用随机重采样正样本
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]

if len(pos_idx) > 0:
    # 复制正样本
    oversample_factor = 3
    oversampled_pos_idx = np.repeat(pos_idx, oversample_factor)
    final_idx = np.concatenate([oversampled_pos_idx, neg_idx])

    X_text_train_res = X_text_train[final_idx]
    X_audio_train_res = [X_audio_train[i] for i in final_idx]
    y_train_res = y_train[final_idx]

    print(f"过采样后训练集正样本数: {np.sum(y_train_res == 1)}")
    print(f"训练集总样本数: {len(y_train_res)}")
else:
    X_text_train_res = X_text_train
    X_audio_train_res = X_audio_train
    y_train_res = y_train


# ========== 4. 创建数据集类 ==========
class FusionDataset(Dataset):
    def __init__(self, text_data, audio_data, labels):
        self.text_data = torch.FloatTensor(text_data)
        self.audio_data = audio_data  # list of numpy arrays
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_data[idx], self.audio_data[idx], self.labels[idx]


def collate_fn(batch):
    """处理变长音频序列（每个样本 3 个音频，拼接成 1 个）"""
    text_batch, audio_batch, label_batch = zip(*batch)

    # 文本直接堆叠
    text_batch = torch.stack(text_batch)

    # 把每个样本的 3 个音频拼接成一个
    merged_audio = []
    merged_lengths = []
    for sample_audios in audio_batch:
        # sample_audios: [audio1, audio2, audio3]，每个是 (T, 128)
        concat_audio = np.concatenate(sample_audios, axis=0)  # (T1+T2+T3, 128)
        merged_audio.append(concat_audio)
        merged_lengths.append(concat_audio.shape[0])

    # Padding
    max_len = max(merged_lengths)
    padded_audio = torch.zeros(len(merged_audio), max_len, merged_audio[0].shape[1])
    for i, a in enumerate(merged_audio):
        padded_audio[i, :a.shape[0], :] = torch.FloatTensor(a)

    label_batch = torch.tensor(label_batch)
    return text_batch, padded_audio, label_batch, merged_lengths


# 创建数据集
train_dataset = FusionDataset(X_text_train_res, X_audio_train_res, y_train_res)
test_dataset = FusionDataset(X_text_test, X_audio_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# ========== 5. 文本编码器 ==========
class TextEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, dropout=0.5):
        super(TextEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.dropout(context)  # (batch, hidden_dim*2=512)


# ========== 6. 音频编码器（支持变长） ==========
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, dropout=0.5):
        super(AudioEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        # 用 pack_padded_sequence 处理变长
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        # 解包
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # 取每个序列的最后一个有效时间步
        last_out = torch.stack([gru_out[i, lengths[i] - 1, :] for i in range(len(lengths))])
        last_out = self.proj(last_out)
        return self.dropout(last_out)


# ========== 7. 融合模型 ==========
class FusionModel(nn.Module):
    def __init__(self, text_dim=1024, audio_dim=128, hidden_dim=256, num_classes=2, dropout=0.5):
        super(FusionModel, self).__init__()
        self.text_encoder = TextEncoder(text_dim, hidden_dim, dropout)
        self.audio_encoder = AudioEncoder(audio_dim, hidden_dim, dropout)

        feat_dim = hidden_dim * 2  # 512
        concat_dim = feat_dim * 2  # 1024

        self.modal_attention = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + concat_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, text_x, audio_x, audio_lengths):
        text_feat = self.text_encoder(text_x)
        audio_feat = self.audio_encoder(audio_x, audio_lengths)

        concat_feat = torch.cat([text_feat, audio_feat], dim=1)
        modal_weights = F.softmax(self.modal_attention(concat_feat), dim=1)
        fused_feat = modal_weights[:, 0:1] * text_feat + modal_weights[:, 1:2] * audio_feat

        output = self.classifier(torch.cat([fused_feat, concat_feat], dim=1))
        return output


# ========== 8. 训练配置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionModel(
    text_dim=1024,
    audio_dim=128,
    hidden_dim=256,
    num_classes=2,
    dropout=0.5
).to(device)

class_weights = torch.FloatTensor([1.0, 2.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

# ========== 9. 训练 ==========
print("\n开始训练...")
best_f1 = 0
best_recall = 0

for epoch in range(150):
    model.train()
    total_loss = 0
    correct = 0
    for text_batch, audio_batch, y_batch, audio_lengths in train_loader:
        text_batch = text_batch.to(device)
        audio_batch = audio_batch.to(device)
        y_batch = y_batch.to(device)
        audio_lengths = audio_lengths

        optimizer.zero_grad()
        output = model(text_batch, audio_batch, audio_lengths)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y_batch).sum().item()

    train_acc = correct / len(train_dataset)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for text_batch, audio_batch, y_batch, audio_lengths in test_loader:
            text_batch = text_batch.to(device)
            audio_batch = audio_batch.to(device)
            output = model(text_batch, audio_batch, audio_lengths)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    from sklearn.metrics import f1_score, recall_score

    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='binary')

    scheduler.step(test_f1)

    if (epoch + 1) % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:3d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test F1: {test_f1:.4f} | Recall: {test_recall:.4f} | LR: {current_lr:.2e}")

    if test_f1 > best_f1:
        best_f1 = test_f1
        best_recall = test_recall
        os.makedirs(os.path.join(prefix, 'Model/ClassificationWhole/Fuse'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(prefix, 'Model/ClassificationWhole/Fuse/fusion_model_balanced.pt'))

# ========== 10. 最终评估 ==========
print("\n加载最佳模型...")
model.load_state_dict(torch.load(os.path.join(prefix, 'Model/ClassificationWhole/Fuse/fusion_model_balanced.pt')))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for text_batch, audio_batch, y_batch, audio_lengths in test_loader:
        text_batch = text_batch.to(device)
        audio_batch = audio_batch.to(device)
        output = model(text_batch, audio_batch, audio_lengths)
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y_batch.numpy())

print("\n最终测试结果:")
print(classification_report(all_labels, all_preds, target_names=['非抑郁', '抑郁症']))
print("混淆矩阵:")
print(confusion_matrix(all_labels, all_preds))
print(f"最佳 F1: {best_f1:.4f}, 最佳召回率: {best_recall:.4f}")
print("模型已保存")