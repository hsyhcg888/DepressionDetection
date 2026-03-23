# \# 基于多模态情感认知的抑郁症检测系统

# 

# \## 项目简介

# 针对抑郁症早期筛查问题，复现 ICASSP 2022 论文，构建文本-音频多模态检测系统。在 EATD-Corpus 数据集上实现抑郁倾向识别，对比传统机器学习与深度学习方法。

# 

# \## 技术栈

# \- \*\*语言\*\*：Python 3.11

# \- \*\*深度学习\*\*：PyTorch 1.8, TensorFlow 2.13

# \- \*\*文本特征\*\*：ELMo (ELMoForManyLangs)

# \- \*\*音频特征\*\*：VGGish

# \- \*\*数据处理\*\*：NumPy, Pandas, Librosa, jieba, scikit-learn

# 

# \## 项目结构

DepressionDetection/

├── README.md

├── requirements.txt

├── text\_features\_whole.py # 文本特征提取 (ELMo)

├── text\_bilstm\_cv.py # 文本分类 + 5折交叉验证

├── audio\_features\_whole.py # 音频特征提取 (VGGish + GRU)

├── fuse\_net\_whole.py # 多模态融合 (模态注意力)

├── TextTraditionalClassifiers.py # 传统方法对比 (SVM/RF/MLP)

├── Data/ # 原始数据 (不传GitHub)

├── Features/ # 特征文件 (不传GitHub)

└── Model/ # 保存的模型 (不传GitHub)

\## 实验结果



\### 文本分类 (BiLSTM-Attention)

\- 5折交叉验证

\- 平均召回率：\*\*46.7%\*\*

\- 平均 F1：\*\*0.411\*\*



\### 多模态融合 (文本 + 音频)

\- 模态注意力机制

\- 精确率：\*\*50%\*\*

\- 召回率：\*\*44%\*\*

\- F1：\*\*0.47\*\*



\### 传统方法对比

| 模型 | 召回率 | 精确率 | F1 |

|------|--------|--------|-----|

| 文本 BiLSTM (深度学习) | \*\*46.7%\*\* | 34% | 0.41 |

| 多模态融合 | 44% | \*\*50%\*\* | \*\*0.47\*\* |

| SVM (Linear) | 23.3% | 24.4% | 0.24 |

| Random Forest | 0% | 0% | 0 |

| MLP (3层) | 13.3% | 42.5% | 0.19 |



\## 工程难点与解决方案



| 问题 | 解决方案 |

|------|----------|

| ELMo 与 PyTorch 新版本不兼容 | 手动修改 `highway.py`，删除 `@overrides` 装饰器 |

| 文本文件编码混用 (GBK/UTF-8) | 添加 `try-except` 自动切换编码 |

| VGGish 环境配置 | 下载模型文件到本地，添加路径到 `sys.path` |

| 变长音频序列处理 | 使用 `pack\_padded\_sequence` 和自定义 `collate\_fn` |

| 数据不平衡 (正负样本 1:4.4) | 过采样 (正样本复制3倍) + 类别权重 (1:3) |



\## 运行方式



\### 1. 安装依赖

```bash

pip install -r requirements.txt

2\. 文本特征提取

bash

python text\_features\_whole.py

3\. 文本分类 (5折交叉验证)

bash

python text\_bilstm\_cv.py

4\. 传统方法对比

bash

python TextTraditionalClassifiers.py

5\. 音频特征提取

bash

python audio\_features\_whole.py

6\. 多模态融合

bash

python fuse\_net\_whole.py

数据集

EATD-Corpus (Emotional Audio-Textual Depression Corpus)



162 名志愿者



包含 SDS 抑郁自评量表评分 (≥53 分为抑郁)



每个样本：3个文本回答 + 3个音频回答



项目来源

复现论文：Automatic Depression Detection: An Emotional Audio-Textual Corpus and a GRU/BiLSTM-based Model (ICASSP 2022)

