# In[1]: Import the necessary package
import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

# In[2]: define the configuration

class CFG:
    seed = 42  # Random seed
    model_name = "microsoft/deberta-v3-base" # Name of pretrained models
    sequence_length = 512  # Input sequence length
    epochs = 10 # Training epochs
    batch_size = 32  # Batch size
    scheduler = 'cosine'  # Learning rate scheduler
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v:k for k, v in label2name.items()}
    BASE_PATH = '../dataset'
torch.cuda.manual_seed(CFG.seed)

# In[3]:  load training data  

df = pd.read_csv(f'{CFG.BASE_PATH}/train.csv') 
df = df.sample(frac=0.20)
# Take the first prompt and its associated response
df["prompt"] = df.prompt.map(lambda x: eval(x)[0])
df["response_a"] = df.response_a.map(lambda x: eval(x.replace("null","''"))[0])
df["response_b"] = df.response_b.map(lambda x: eval(x.replace("null", "''"))[0])
df["class_name"] = df[["winner_model_a", "winner_model_b" , "winner_tie"]].idxmax(axis=1)
df["label"] = df.class_name.map(CFG.name2label)
# Show Sample
df.head()

# In[4]: process the raw_data, then get the train and validate dataset

def make_pairs(row):  # define a function to combine the prompt and response.
    row["encode_fail"] = False
    try:
        prompt = row.prompt.encode("utf-8").decode("utf-8")
    except:
        prompt = ""
        row["encode_fail"] = True

    try:
        response_a = row.response_a.encode("utf-8").decode("utf-8")
    except:
        response_a = ""
        row["encode_fail"] = True

    try:
        response_b = row.response_b.encode("utf-8").decode("utf-8")
    except:
        response_b = ""
        row["encode_fail"] = True
        
    row['options'] = [f"Prompt: {prompt}\n\nResponse: {response_a}",  # Response from Model A
                      f"Prompt: {prompt}\n\nResponse: {response_b}"  # Response from Model B
                     ]
    return row

df = df.apply(make_pairs, axis=1)  # Apply the make_pairs function to each row in df
df.encode_fail.value_counts(normalize=False)  
from sklearn.model_selection import train_test_split  

train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df["label"])

# In[5]:  preprocess the data,  then generate the dataloader.

from datasets import Dataset, load_from_disk  
from transformers import AutoTokenizer  
  
MODEL_NAME = 'microsoft/deberta-base'  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  
  
def tokenize_function(examples):  
    return tokenizer(examples['options'], truncation=True, padding='max_length', max_length=CFG.sequence_length)  
  
# 检查是否已存在本地文件  
try:  
    tokenized_train_datasets = load_from_disk('tokenized_train_dataset')  
    tokenized_valid_datasets = load_from_disk('tokenized_valid_dataset')  
    print("Loaded tokenized datasets from disk.")  
except:  
    # 如果不存在，则进行标记化并保存  
    train_dataset = Dataset.from_pandas(train_df[['options', 'label']])  
    valid_dataset = Dataset.from_pandas(valid_df[['options', 'label']])  
  
    tokenized_train_datasets = train_dataset.map(tokenize_function)  
    tokenized_valid_datasets = valid_dataset.map(tokenize_function)  
  
    # 保存到本地  
    tokenized_train_datasets.save_to_disk('tokenized_train_dataset')  
    tokenized_valid_datasets.save_to_disk('tokenized_valid_dataset')  
    print("Tokenized datasets saved to disk.")  

tokenized_train_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
tokenized_valid_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataloader = torch.utils.data.DataLoader(tokenized_train_datasets, batch_size= CFG.batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(tokenized_valid_datasets, batch_size= CFG.batch_size, shuffle=False)

# In[6]: define the model

from transformers import DebertaModel, DebertaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptResponseClassifier(nn.Module):
    def __init__(self, backbone):
        super(PromptResponseClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 2, 3)  # 调整输出维度为3，用于3类分类

    def forward(self, input):
        # 从批量中提取 inputs_id 和 attention_mask
        input_ids_a = input['input_ids'][:, 0, :]
        attention_mask_a = input['attention_mask'][:, 0, :]
        input_ids_b = input['input_ids'][:, 1, :]
        attention_mask_b = input['attention_mask'][:, 1, :]

        # 通过骨干模型
        embed_a = self.backbone(input_ids_a, attention_mask=attention_mask_a)[0]  # [batch_size, seq_len, hidden_size]
        embed_b = self.backbone(input_ids_b, attention_mask=attention_mask_b)[0]  # [batch_size, seq_len, hidden_size]

        # 拼接嵌入
        concatenated_embed = torch.cat((embed_a, embed_b), dim=-1)  # [batch_size, seq_len, hidden_size * 2]

        # 对序列长度进行平均池化
        pooled_output = torch.mean(concatenated_embed, dim=1)  # [batch_size, hidden_size * 2]

        # 应用 dropout 并通过分类器
        outputs = self.classifier(self.dropout(pooled_output))
        return outputs
    
# In[7]: iniitialiize the model

configuration = DebertaConfig()
backbone = DebertaModel.from_pretrained('microsoft/deberta-base')
model = PromptResponseClassifier(backbone)

# In[8]:

import torch  
import torch.optim as optim  
from torch.cuda.amp import autocast, GradScaler  
import wandb  
  
# 初始化 wandb  
wandb.init(project="preference_classification")  
  
# 检查 GPU 是否可用  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using device: {device}")  
  
# 将模型移动到 GPU  
model.to(device)  
  
# 定义损失函数和优化器  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=5e-6)  
  
# 混合精度训练的 scaler  
scaler = GradScaler()  


def evaluate(model, dataloader, device):  
    model.eval()  
    correct = 0  
    total = 0  
    with torch.no_grad():  
        for batch in dataloader:  
            input_ids = batch['input_ids'].to(device)  
            attention_mask = batch['attention_mask'].to(device)  
            labels = batch['label'].to(device)  
  
            outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})  
            _, predicted = torch.max(outputs, 1)  
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  
      
    accuracy = correct / total  
    return accuracy  


# 训练循环  
for epoch in range(CFG.epochs):  
    model.train()  
    total_loss = 0  
    correct = 0  
    total = 0  
    for batch_idx, batch in enumerate(train_dataloader):  
        # 将数据移动到 GPU  
        input_ids = batch['input_ids'].to(device)  
        attention_mask = batch['attention_mask'].to(device)  
        labels = batch['label'].to(device)  
  
        with autocast():  # 混合精度  
            # 前向传播  
            outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})  
            loss = criterion(outputs, labels)  
  
        # 计算准确性  
        _, predicted = torch.max(outputs, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()  
  
        total_loss += loss.item()  
  
        # 反向传播和优化  
        optimizer.zero_grad()  
        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update()  
  
        # 记录到 wandb  
        wandb.log({  
            "Batch Loss": loss.item(),  
            "Batch Accuracy": correct / total,  
            "Epoch": epoch + 1  
        })  
  
    avg_loss = total_loss / len(train_dataloader)  
    accuracy = correct / total  
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')  
    
     # 在验证集上评估  
    val_accuracy = evaluate(model, valid_dataloader, device)  
    print(f'Validation Accuracy: {val_accuracy:.4f}')  

    # 记录 epoch 级别的指标到 wandb  
    wandb.log({  
        "Epoch Loss": avg_loss,  
        "Epoch Accuracy": accuracy,  
        "Validation Accuracy": val_accuracy, 
        "Epoch": epoch + 1  
    })  
  
wandb.finish()  