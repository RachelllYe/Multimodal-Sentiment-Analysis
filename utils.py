import os
import torch
import argparse
from configs import add_args
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset

parser = argparse.ArgumentParser()
args = add_args(parser)

def get_img_text(data):
    img = []
    text = []
    for item in data:
        text.append(item[0])
        img.append(item[1])
    return text,img

def get_files_name(args):
    data_dir = args.work_dir+'/data'
    files = os.listdir(data_dir)
    return files
        
def build_id_2_label(args):
    label_path = args.work_dir+'/label.txt'
    with open(label_path,'r',encoding='gb2312') as f :
        context = f.readlines()

    del context[0]
    id_2_label = {}
    label_2_idx = {'negative':0,'neutral':1,'positive':2}

    for item in context:
        temp = item.strip().split(',')
        id_2_label[temp[0]] = label_2_idx[temp[1]]
    # print(id_2_label)
    # 从小到大
    id_2_label_sorted = {k:v for k, v in sorted(id_2_label.items(),key = lambda x:x[0])}
    return id_2_label_sorted

def build_idx_2_data(args):
    files = get_files_name(args)
    # print(files)
    # files = ['3644.txt','1053.txt','1047.txt']
    data_dir = args.work_dir+'/data'
    train_idx_2_label = build_id_2_label(args)
    y = list(train_idx_2_label.values())
    #print(y)
    # y =[0,1,2]
    train_idx = list(train_idx_2_label.keys())
    # train_idx = ['406','407','408']
    
    train_text = {}
    test_text = {}
    train_img = {}
    test_img = {}
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )
    for file in files:
        file_path = data_dir + "/"+file
        if file[-4:] == ".txt":
            idx = file[:-4]
            with open(file_path,'r',encoding='gb18030') as f :
                context = f.read().strip()
                if idx in train_idx:
                    train_text[idx] = context
                else:
                    test_text[idx] = context
        elif file[-4:] == ".jpg":
            idx = file[:-4]
            img = Image.open(file_path)
            if idx in train_idx:
                train_img[idx] = transform(img)
                # print(train_img[idx].shape)
                # train_img[idx] = img
            else:
                test_img[idx]=transform(img)
                # test_img[idx]=img
    # 一一对齐
    train_text_sorted = {k:v for k, v in sorted(train_text.items(),key = lambda x:x[0])}
    test_text_sorted = {k:v for k, v in sorted(test_text.items(),key = lambda x:x[0])}
    train_img_sorted = {k:v for k, v in sorted(train_img.items(),key = lambda x:x[0])}
    test_img_sorted =  {k:v for k, v in sorted(test_img.items(),key = lambda x:x[0])}
    
    train_data = list(zip(list(train_text_sorted.values()),list(train_img_sorted.values())))
    # test_data = zip(list(test_text_sorted.values()),list(test_img_sorted.values()))
    X_train, X_val, y_train, y_val = \
    train_test_split(train_data, y, test_size=0.25)
    train_text, train_img = get_img_text(X_train)
    val_text,val_img = get_img_text(X_val)
    test_text = list(test_text_sorted.values())
    test_img = list(test_img_sorted.values())
    train_label = y_train
    val_label = y_val
    test_id = list(test_text_sorted.keys())
    for i in range(len(test_id)):
        test_id[i] = int(test_id[i])
    return train_text,train_img,val_text,val_img,test_text,test_img,train_label,val_label,test_id


def preprocessing_for_bert(data,tokenizer,max_len):
    # print(data.shape)
    # 空列表来储存信息
    input_ids = []
    attention_masks = []
    token_type_ids = []

    # 每个句子循环一次
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # 预处理语句
            add_special_tokens=True,  # 加 [CLS] 和 [SEP]
            max_length=max_len,  # 截断或者填充的最大长度
            padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
            return_attention_mask=True  # 返回 attention mask
        )

        # 把输出加到列表里面
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        token_type_ids.append(encoded_sent.get('token_type_ids'))
    # 把list转换为tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    token_type_ids = torch.tensor(token_type_ids)
    return input_ids, attention_masks,token_type_ids

preds = {3650: 'neutral', 2566: 'positive', 2572: 'neutral'}
def write_reults_to_file(preds,args):
    # preds 为字典 {id:res}
    with open(args.test_file, 'r', encoding='utf-8') as f:
        text = f.readlines()
    del text[0]
    
    # text = ['1,null\n','2,null\n']
    for i in range(len(text)):
        item = text[i].split(",")
        id = item[0]
        text[i] = id + ','+preds[int(id)]+"\n"
    res = ''.join(text)
    f = open(args.result_path,mode='w')
    f.write("guid,tag\n")
    f.write(res)
    f.close()
# write_reults_to_file(preds,args)

class MyDataset(Dataset):
    def __init__(self,inputs,masks,tokens,imgs,labels = None,test_id = None) -> None:
        super().__init__()
        self.inputs = inputs
        self.masks = masks
        self.labels = labels
        self.tokens = tokens
        self.imgs = imgs
        self.test_id = test_id

    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # 测试集
        if self.labels == None:
            return self.inputs[index],self.masks[index],self.tokens[index],self.imgs[index],self.test_id[index]
        else:
            return self.inputs[index],self.masks[index],self.tokens[index],self.imgs[index],self.labels[index]




