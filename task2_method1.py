from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from torch.nn import NLLLoss
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, BertModel
import torch
import numpy as np
import os
from convert_offset import convert_offset_and_detect_errorsentences
import pickle
import itertools
import torch.nn as nn
import torch.nn.functional as F
import sys

args = sys.argv

max_length = 256
class Linear(nn.Module):
        def __init__(self,embed_dim, num_class, batch_size):
                super().__init__()
                # self.bn1 = nn.BatchNorm1d(batch_size)
                self.fc1 = nn.Linear(embed_dim, num_class)
                # self.bn2 = nn.BatchNorm1d(batch_size)
                self.fc2 = nn.Linear(embed_dim, num_class)
                # self.bn3 = nn.BatchNorm1d(batch_size)
                self.fc3 = nn.Linear(embed_dim, num_class)
                # self.bn4 = nn.BatchNorm1d(batch_size)
                self.fc4 = nn.Linear(embed_dim, num_class)
                self.init_weights()

        def init_weights(self):
                initrange = 0.5
                # self.embedding.weight.data.uniform_(-initrange, initrange)
                self.fc1.weight.data.uniform_(-initrange, initrange)
                self.fc1.bias.data.zero_()
                self.fc2.weight.data.uniform_(-initrange, initrange)
                self.fc2.bias.data.zero_()
                self.fc3.weight.data.uniform_(-initrange, initrange)
                self.fc3.bias.data.zero_()
                self.fc4.weight.data.uniform_(-initrange, initrange)
                self.fc4.bias.data.zero_()

        def forward(self, embedded):
                embedded1 = self.fc1(embedded)
                # embedded1 = self.bn1(embedded1)
                embedded2 = self.fc2(embedded)
                # embedded2 = self.bn2(embedded2)
                embedded3 = self.fc3(embedded)
                # embedded3 = self.bn3(embedded3)
                embedded4 = self.fc4(embedded)
                # embedded4 = self.bn4(embedded4)
                return embedded1,embedded2,embedded3,embedded4
# GPUが使えれば利用する設定

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('loading model.....')
model_dir = './model_save/'

batch_size = 16
######################################## SET MODELS you wanna test #############################################
pretrained_data_name_list = ['bert-base-uncased']
epoch_list = [88]
######################################## SET MODELS you wanna test #############################################

for pretrained_data_name in pretrained_data_name_list:
        for epoch in epoch_list:
                model_PATH = model_PATH ='{}model_{}_b_{}_e_{}'.format(model_dir, pretrained_data_name, batch_size, epoch)
                # model = BertModel.from_pretrained(
                #         pretrained_data_name,  
                #         num_labels=2,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
                #         output_attentions=False,  # アテンションベクトルを出力するか
                #         output_hidden_states=False,  # 隠れ層を出力するか
                #         )
                model_state_dict = torch.load(model_PATH)
                model = BertModel.from_pretrained(
                        pretrained_data_name,  
                        num_labels=max_length,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
                        output_attentions=False,  # アテンションベクトルを出力するか
                        output_hidden_states=False,  # 隠れ層を出力するか
                        state_dict=model_state_dict
                        )
                model.to(device)
                
                # classifier_PATH = model_PATH ='{}cllasifier_{}_b_{}_e_{}'.format(model_dir, pretrained_data_name, batch_size, epoch)
                # model_state_dict = torch.load(classifier_PATH)
                # model_classifier = Linear(898, 768, 251, batch_size, state_dict=model_state_dict)
                # model_classifier.to(device)
                
                # model_PATH = model_PATH ='{}model_{}_b_{}_e_{}'.format(model_dir, pretrained_data_name, batch_size, epoch)
                # model = BertModel.from_pretrained(
                #         pretrained_data_name,  
                #         num_labels=2,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
                #         output_attentions=False,  # アテンションベクトルを出力するか
                #         output_hidden_states=False,  # 隠れ層を出力するか
                #         )
                # model.cuda()
                # model.load_state_dict(torch.load(model_PATH))
                classifier_PATH = model_PATH ='{}classifier_{}_b_{}_e_{}'.format(model_dir, pretrained_data_name, batch_size, epoch)
                model_classifier = Linear(768, max_length, batch_size)
                model_classifier.cuda()
                model_classifier.load_state_dict(torch.load(classifier_PATH))

                print('loading test.csv.....')
                f_test = "./2020NLP/data/train.csv"
                df_test = pd.read_csv(f_test, delimiter=';', dtype=object)

                Index = df_test.Index.values
                Text = df_test.Text.values
                # Cause = df_train.Cause.values
                # Effect = df_train.Effect.values
                # Cause_Start = df_train.Cause_Start.values
                # Cause_End = df_train.Cause_End.values
                # Effect_Start = df_train.Effect_Start.values
                # Effect_End = df_train.Effect_End.values
                # Sentence = df_train.Sentence.values
                tokenizer = BertTokenizer.from_pretrained(pretrained_data_name)
                input_ids = []
                attention_masks = []
                # labels = []
                for i,t in enumerate(Text):
                        # if i in error_sentence:
                        #         continue
                        encoded_dict = tokenizer.encode_plus(
                                t,
                                add_special_tokens=True,  # Special Tokenの追加
                                max_length=max_length,           # 文章の長さを固定（Padding/Trancatinating）
                                pad_to_max_length=True,  # PADDINGで埋める
                                return_attention_mask=True,   # Attention maksの作成
                                return_tensors='pt',  # Pytorch tensorsで返す
                        )
                        # temp_max = max(encoded_dict['input_ids'][0])
                        # if temp_max > max_id:
                        #         max_id = temp_max
                        input_ids.append(encoded_dict['input_ids'])
                        attention_masks.append(encoded_dict['attention_mask'])
                        # labels.append([Index_Cause_token_start[i],Index_Cause_token_end[i],Index_Effect_token_start[i],Index_Effect_token_end[i]])
                input_ids = torch.cat(input_ids, dim=0)
                attention_masks = torch.cat(attention_masks, dim=0)
                # labels = torch.tensor(labels)
                dataset = TensorDataset(input_ids, attention_masks)
                dataloader = DataLoader(
                        # val_dataset,
                        dataset,
                        # sampler=SequentialSampler(val_dataset),  # 順番にデータを取得してバッチ化
                        sampler=SequentialSampler(dataset),  # 順番にデータを取得してバッチ化
                        batch_size = batch_size
                        )
                # labels = torch.tensor(labels_train)
                print('predicting w/ testset.....')
                model.eval()  # 訓練モードをオフ
                # val_loss = 0
                # df = pd.DataFrame()
                pred1=[]
                pred2=[]
                pred3=[]
                pred4=[]
                with torch.no_grad():  # 勾配を計算しない
                        for i, batch in enumerate(dataloader):
                                b_input_ids = batch[0].to(device)
                                b_input_mask = batch[1].to(device)
                                # b_labels = batch[2].to(device)
                                with torch.no_grad():
                                        logits_ = model(b_input_ids,
                                                        token_type_ids=None,
                                                        attention_mask=b_input_mask)

                                        logit1,logit2,logit3,logit4 = model_classifier(logits_[1])
                                        pred1.append(np.argmax(logit1.cpu().numpy(), axis=1))
                                        pred2.append(np.argmax(logit2.cpu().numpy(), axis=1))
                                        pred3.append(np.argmax(logit3.cpu().numpy(), axis=1))
                                        pred4.append(np.argmax(logit4.cpu().numpy(), axis=1))
                
                # FOR SAVING
                pred1 = list(itertools.chain.from_iterable(pred1))
                pred2 = list(itertools.chain.from_iterable(pred2))
                pred3 = list(itertools.chain.from_iterable(pred3))
                pred4 = list(itertools.chain.from_iterable(pred4))

                pred1_df = pd.DataFrame(pred1,columns=['pred_CS'])
                pred2_df = pd.DataFrame(pred2,columns=['pred_CE'])
                pred3_df = pd.DataFrame(pred3,columns=['pred_ES'])
                pred4_df = pd.DataFrame(pred4,columns=['pred_EE'])
                indices_df = pd.concat([pred1_df,pred2_df,pred3_df,pred4_df], axis=1)
                # print(indices_df)
                Index_df = pd.DataFrame(Index, columns=['Index'])
                Text_df = pd.DataFrame(Text, columns=['Text'])
                Tokens=[]
                for sen in Text:
                        token_words = tokenizer.tokenize(str(sen))
                        Tokens.append(token_words)

                Cause = []
                Effect = []
                for i in range(len(Tokens)):
                        # print(len(Tokens))
                        # print(len(pred1))
                        if pred1[i]>pred2[i]:
                                Cause.append(" ".join(Tokens[i][pred2[i]:pred1[i]]))
                        else:
                                Cause.append(" ".join(Tokens[i][pred1[i]:pred2[i]]))
                        if pred3[i]>pred4[i]:
                                Effect.append(" ".join(Tokens[i][pred4[i]:pred3[i]]))
                        else:
                                Effect.append(" ".join(Tokens[i][pred3[i]:pred4[i]]))
                Cause_df = pd.DataFrame(Cause, columns=['Cause'])
                Effect_df = pd.DataFrame(Effect, columns=['Effect'])
                output_df = pd.concat([Index_df,Text_df,Cause_df,Effect_df], axis=1)
 
                # output_dir = './submission_save/'
                # output_PATH ='{}{}_b_{}_e_{}.csv'.format(output_dir, pretrained_data_name, batch_size, epoch)
                # if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                output_PATH = args[1]
                output_df.to_csv(output_PATH, sep=';',index=False)
                print('Saved {}'.format(output_PATH))
                        