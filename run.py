from pickletools import optimize
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import build_id_2_label,build_idx_2_data,preprocessing_for_bert,MyDataset,write_reults_to_file
from transformers import BertTokenizer
from configs import add_args
from pytorch_pretrained_bert import BertAdam
import argparse
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from model import MultimodalConcatBertClf

parser = argparse.ArgumentParser()
args = add_args(parser)

args.work_dir = os.getcwd()
args.result_path = args.work_dir + args.result_path
args.test_file = args.work_dir + args.test_file

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

train_text,train_img,val_text,val_img,test_text,test_img,train_label,val_label,test_id=build_idx_2_data(args)
# print(len(test_id))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# print(len(test_text))
X = []
X.extend(train_text)
X.extend(val_text)
X.extend(test_text)

encoded_comment = [tokenizer.encode(sent, add_special_tokens=True) for sent in X]
# 找到句子的最大长度
max_len = max([len(sent) for sent in encoded_comment])

# 返回结果为tensor
train_inputs, train_masks, train_token_type_ids = preprocessing_for_bert(train_text,tokenizer,max_len)
val_inputs, val_masks, val_token_type_ids = preprocessing_for_bert(val_text,tokenizer,max_len)
test_inputs, test_masks, test_token_type_ids = preprocessing_for_bert(test_text,tokenizer,max_len)

# print(train_inputs.shape)
train_labels = torch.tensor(train_label)
val_labels = torch.tensor(val_label)
test_id = torch.tensor(test_id)

# 给训练集创建 DataLoader
# train_data = TensorDataset(train_inputs, train_masks, train_labels,train_token_type_ids,train_img)
train_data = MyDataset(inputs = train_inputs,masks = train_masks,tokens = train_token_type_ids,imgs = train_img,labels = train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
# print(train_dataloader)

# 给验证集创建 DataLoader
# val_data = TensorDataset(val_inputs, val_masks, val_labels,val_token_type_ids,val_img)
val_data = MyDataset(inputs = val_inputs,masks = val_masks,tokens = val_token_type_ids,imgs = val_img,labels = val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

# test_data = TensorDataset(test_inputs, test_masks, val_token_type_ids,test_id,test_img)
test_data = MyDataset(inputs=test_inputs,masks = test_masks,tokens = test_token_type_ids,imgs = test_img,test_id = test_id)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

model = MultimodalConcatBertClf(args)
# 用GPU运算
model.to(device)
# 创建优化器
optimizer = AdamW(model.parameters(),
                    lr=5e-5,  # 默认学习率
                    eps=1e-8  # 默认精度
                    )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
# 训练的总步数
total_steps = len(train_dataloader) * args.epochs
# 学习率预热
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value
                                            num_training_steps=total_steps)

loss_fn = nn.CrossEntropyLoss()  # 交叉熵
# 训练模型
def train(model, train_dataloader, val_dataloader=None, epochs=2, evaluation=False):
    # 表头
    print(f"{'Epoch':^7} | {'训练集 Loss':^12} | {'训练集准确率':^9} | {'验证集 Loss':^10} | {'验证集准确率':^9} |")
    print("-" * 80)
    # 开始训练循环
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        total_loss, batch_counts,total_acc = 0.0, 0,0

        # 把model放到训练模式
        model.train()

        # 分batch训练
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # 把batch加载到GPU
            b_input_ids, b_attn_mask, b_token,b_img,b_labels= tuple(t.to(device) for t in batch)
            #print(b_labels.shape)
            # 归零导数
            model.zero_grad()
            # 真正的训练
            logits = model(b_input_ids,b_attn_mask, b_token,b_img)
            #print(logits.shape)
            # 计算loss并且累加

            loss = loss_fn(logits, b_labels)

            # get预测结果，这里就是求每行最大的索引咯，然后用flatten打平成一维
            preds = torch.argmax(logits, dim=1).flatten()  

            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            total_acc += accuracy
            # batch_loss += loss.item()
            total_loss += loss.item()
            # 反向传播
            loss.backward()
            # 归一化，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数和学习率
            optimizer.step()
            scheduler.step()

        # 计算平均loss 这个是训练集的loss
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:  
            # 在我们的验证集/测试集上.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            print(
                f"{epoch_i + 1:^7} | {avg_train_loss:^15.6f} | {avg_train_acc:^14.2f}% | {val_loss:^13.6f} | {val_accuracy:^14.2f}% |")
            print("-" * 80)


# 在测试集上面来看看我们的训练效果
def evaluate(model, val_dataloader):
    """
    在每个epoch后验证集上评估model性能
    """
    # model放入评估模式
    model.eval()

    # 准确率和误差
    val_accuracy = []
    val_loss = []

    # 验证集上的每个batch
    for batch in val_dataloader:
        # 放到GPU上
        b_input_ids, b_attn_mask, b_token,b_img,b_labels = tuple(t.to(device) for t in batch)

        # 计算结果，不计算梯度
        with torch.no_grad():
            logits = model(b_input_ids,b_attn_mask, b_token,b_img)  
            
        loss = loss_fn(logits, b_labels.long())
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()  # 返回一行中最大值的序号

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # 计算整体的平均正确率和loss
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def test(model, test_dataloader):
    idx_2_label = {0:'negative',1:'neutral',2:'positive'}
    # label_2_idx = {'negative':0,'neutral':1,'positive':2}
    pred_dict = {}
    # print("before")
    model.eval()
    # print("after")
    for batch in test_dataloader:
        b_input_ids, b_attn_mask,b_token,b_img,b_id = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask,b_token,b_img)
        preds = torch.argmax(logits, dim=1).flatten()
        preds = preds.tolist()
        b_id = b_id.tolist()
        for i in range(len(preds)):
            pred_dict[b_id[i]] = idx_2_label[preds[i]]
    return pred_dict

# print("Start training and validation:\n")
# print("Start training and testing:\n")
train(model,train_dataloader, val_dataloader, epochs=args.epochs, evaluation=True)  # 这个是有评估的
preds  = test(model,test_dataloader)
write_reults_to_file(preds,args)


