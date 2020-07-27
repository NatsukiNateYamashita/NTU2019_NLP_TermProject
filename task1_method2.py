from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
import torch
import numpy as np
import os
import itertools
import sys

args = sys.argv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##### PREDICT

##### LOAD
print('loading model.....')
model_dir = './model_save/'
batch_size = 16
######################################## SET MODELS you wanna test #############################################
pretrained_data_name_list = ['bert-base-uncased']
epoch_list = [4]
# pretrained_data_name_list = ['bert-base-cased']
# epoch_list = [23,24,25,26,27,28,29]
######################################## SET MODELS you wanna test #############################################

for pretrained_data_name in pretrained_data_name_list:
        for epoch in epoch_list:
                model_PATH = model_PATH ='{}{}_b_{}_e_{}'.format(model_dir, pretrained_data_name, batch_size, epoch)

                model = BertForSequenceClassification.from_pretrained(
                        pretrained_data_name,  
                        num_labels=2,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
                        output_attentions=False,  # アテンションベクトルを出力するか
                        output_hidden_states=False,  # 隠れ層を出力するか
                        )
                # Use GPU
                model.cuda()    
                model.load_state_dict(torch.load(model_PATH))


                print('loading test.csv.....')
                f_test = "./nlp-class-2020-fincausal-task1/test.csv"
                df_test = pd.read_csv(f_test, delimiter=';')
                sentenses_test = df_test.Text.values
                index_test = df_test.Index.values

                print('tokenizing.....')
                tokenizer = BertTokenizer.from_pretrained(pretrained_data_name)
                input_ids_test = []
                attention_masks_test = []
                for sen in sentenses_test:
                        encoded_dict = tokenizer.encode_plus(
                                str(sen),
                                add_special_tokens=True,  # Special Tokenの追加
                                max_length=512,           # 文章の長さを固定（Padding/Trancatinating）
                                pad_to_max_length=True,  # PADDINGで埋める
                                return_attention_mask=True,   # Attention maksの作成
                                return_tensors='pt',  # Pytorch tensorsで返す
                                )

                        input_ids_test.append(encoded_dict['input_ids'])
                        attention_masks_test.append(encoded_dict['attention_mask'])
                input_ids_test = torch.cat(input_ids_test, dim=0)
                attention_masks_test = torch.cat(attention_masks_test, dim=0)
                dataset = TensorDataset(input_ids_test, attention_masks_test)
                batch_size = 16
                test_dataloader = DataLoader(
                        # val_dataset,
                        dataset,
                        # sampler=SequentialSampler(val_dataset),  # 順番にデータを取得してバッチ化
                        sampler=SequentialSampler(dataset),  # 順番にデータを取得してバッチ化
                        batch_size = batch_size
                        )
                # labels = torch.tensor(labels_train)
                print('predicting w/ testset.....')
                
                model.eval()  # 訓練モードをオフ
                pred_con_df = pd.DataFrame()
                with torch.no_grad():  # 勾配を計算しない
                        for i, batch in enumerate(test_dataloader):
                                # print('##########################################################',i)
                                b_input_ids = batch[0].to(device)
                                b_input_mask = batch[1].to(device)
                                # print(b_input_ids.shape)
                                preds = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask)
                                logits_df = pd.DataFrame(preds[0].cpu().numpy(),columns=['logit_0', 'logit_1'])
                                # np.argmaxで大き方の値を取得
                                # pred_df = pd.DataFrame(np.argmax(preds[0].cpu().numpy(), axis=1), columns=['Gold'])
                                pred_df = pd.DataFrame(np.argmax(preds[0].cpu().numpy(), axis=1))
                                # print(pred_df)
                                # index_df = pd.DataFrame(index_test.cpu().numpy(), columns=['Index'])
                                # index_df = pd.DataFrame(index_test, columns=['Index'])
                                # index_df = pd.DataFrame(index_test)
                                # temp_df = pd.concat([index_df, pred_df], axis=1)
                                # print(temp_df)
                                # output_df = pd.concat([output_df, temp_df])
                                pred_con_df = pd.concat([pred_con_df, pred_df])
                        index_df = pd.DataFrame(index_test)
                        index_df.columns = ['Index']
                        # pred_con_df.columns = ['Gold']

                        # print(index_df, pred_con_df)       

                        # output_df = pd.concat([index_df, pred_con_df],)
                        output_df = index_df
                        output_df['Gold'] = list(itertools.chain.from_iterable(pred_con_df.values.tolist()))
                        # print(output_df)
                        # print(pred_con_df.values.tolist())
                        # output_df.columns = ['Index','Gold']

                                
                # output_dir = './submission_save/'
                # output_PATH ='{}{}_b_{}_e_{}.csv'.format(output_dir, pretrained_data_name, batch_size, epoch)
                output_PATH = args[1]
                # if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                output_df.to_csv(output_PATH,index=False)
                print('Saved {}'.format(output_PATH))
