import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

# ========== 1. 加载特征和标签 ==========
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

# ========== 2. 数据标准化 ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(text_features)

# ========== 3. 定义模型 ==========
models = {
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
    'SVM (Linear)': SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
    'MLP (2层)': MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                               solver='adam', max_iter=500, random_state=42),
    'MLP (3层)': MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu',
                               solver='adam', max_iter=500, random_state=42),
}

# ========== 4. 5折交叉验证 ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("5 折交叉验证结果:")
print("-" * 60)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 60)

results = {}

for name, model in models.items():
    # 交叉验证
    acc_scores = cross_val_score(model, X_scaled, text_targets, cv=cv, scoring='accuracy')
    prec_scores = cross_val_score(model, X_scaled, text_targets, cv=cv, scoring='precision')
    rec_scores = cross_val_score(model, X_scaled, text_targets, cv=cv, scoring='recall')
    f1_scores = cross_val_score(model, X_scaled, text_targets, cv=cv, scoring='f1')

    results[name] = {
        'accuracy': acc_scores.mean(),
        'precision': prec_scores.mean(),
        'recall': rec_scores.mean(),
        'f1': f1_scores.mean(),
    }

    print(f"{name:<20} {acc_scores.mean():<12.4f} {prec_scores.mean():<12.4f} "
          f"{rec_scores.mean():<12.4f} {f1_scores.mean():<12.4f}")

print("-" * 60)

# ========== 5. 训练最佳模型并显示详细报告 ==========
print("\n" + "=" * 60)
print("最佳模型详细结果 (Random Forest):")
print("=" * 60)

best_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)

for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, text_targets)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = text_targets[train_idx], text_targets[test_idx]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\nFold {fold + 1}:")
    print(classification_report(y_test, y_pred, target_names=['非抑郁', '抑郁症']))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\n训练完成！")