from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import pandas as pd
import torch
def convert_offset_and_detect_errorsentences(Pretrained_data_name):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ###### Loading dataset #######
        print('reading data.....')
        f_train = "./2020NLP/data/train.csv"
        df_train = pd.read_csv(f_train, delimiter=';', dtype=object)

        Index = df_train.Index.values
        Text = df_train.Text.values
        Cause = df_train.Cause.values
        Effect = df_train.Effect.values
        Cause_Start = df_train.Cause_Start.values
        Cause_End = df_train.Cause_End.values
        Effect_Start = df_train.Effect_Start.values
        Effect_End = df_train.Effect_End.values
        Sentence = df_train.Sentence.values
        pretrained_data_name = Pretrained_data_name
        tokenizer = BertTokenizer.from_pretrained(pretrained_data_name)

        print("converting offset......")

        Index_Cause_token_start = []
        Index_Cause_token_end = []
        Index_Effect_token_start = []
        Index_Effect_token_end = []

        count_1 = 0
        count_2 = 0
        error_sentence = []
        for i in range(len(Sentence)):
                # print('####### {} th sentence #######'.format(i))
                Sentence_token = tokenizer.tokenize(str(Sentence[i]))
                count_1 = 0
                count_2 = 0
                for index, s in enumerate(Sentence_token):
                        if s == "<" and Sentence_token[index+2] == "##1":
                                count_1 +=1
                        elif s == "<" and Sentence_token[index+2] == "##2":
                                count_2 +=1
                if count_1 > 1 or count_2 >1:
                        error_sentence.append(i)
                # print("error_sentence",error_sentence)
                                

        for i in range(len(Sentence)):
                if i in error_sentence:
                        Index_Cause_token_start.append(0)
                        Index_Cause_token_end.append(0)
                        Index_Effect_token_start.append(0)
                        Index_Effect_token_end.append(0)
                        continue
                print('####### {} th sentence #######'.format(i))
                Sentence_token = tokenizer.tokenize(str(Sentence[i]))
                Text_token = tokenizer.tokenize(str(Text[i]))
                Cause_token = tokenizer.tokenize(str(Cause[i]))
                Effect_token = tokenizer.tokenize(str(Effect[i]))
                print(Sentence_token)
                for index, s in enumerate(Sentence_token):
                        if s == "<" and Sentence_token[index+2] == "##1":
                                Index_Cause_token_start.append(index)
                                del Sentence_token[index:index+4]
                                for ind, s in enumerate(Sentence_token):
                                        if s == "<" and Sentence_token[ind+3] == "##1":
                                                print('##1: ',Sentence_token[ind-1])
                                                print('##1: ',ind-1)
                                                Index_Cause_token_end.append(ind-1)
                                                print('##1: ',Index_Cause_token_end[i])
                                                print('##1: ',Sentence_token[Index_Cause_token_end[i]:Index_Cause_token_end[i]+3])
                                                del Sentence_token[ind:ind+5]
                                                print('##1: ',Sentence_token)
                                                break
                        elif s == "<" and Sentence_token[index+1] == "e" and Sentence_token[index+2] == "##2":
                                Index_Effect_token_start.append(index)
                                del Sentence_token[index:index+4]
                                for ind, s in enumerate(Sentence_token):
                                        if s == "<" and Sentence_token[ind+2] == "e" and Sentence_token[ind+3] == "##2":
                                                print('##2: ',Sentence_token[ind-1])
                                                print('##2: ',ind-1)
                                                Index_Effect_token_end.append(ind-1)
                                                print('##2: ',Index_Effect_token_end[i])
                                                print('##2: ',Sentence_token[Index_Effect_token_end[i]:Index_Effect_token_end[i]+3])
                                                del Sentence_token[ind:ind+5]
                                                print('##2: ',Sentence_token)
                                                # print(Sentence_token[Index_Effect_token_end[i]:Index_Effect_token_end[i]+3])
                                                break
                try:
                        print('Text: ',Text[i])
                        print('Sent: '," ".join(Sentence_token))
                        print("Text len == Sent len: {} == {}".format(len(Text_token),len(Sentence_token)))
                        print('Cause: ',Cause[i])
                        print('Effect: ',Effect[i])
                        print('Cause_token[0] == Text_token[Index]   :  {} == {}'.format(Cause_token[0], Text_token[Index_Cause_token_start[i]]))
                        print('Cause_token[-1] == Text_token[Index]  :  {} == {}'.format(Cause_token[-1], Text_token[Index_Cause_token_end[i]]))
                        print('Effect_token[0] == Text_token[Index]   :  {} == {}'.format(Effect_token[0], Text_token[Index_Effect_token_start[i]]))
                        print('Effect_token[-1] == Text_token[Index]  :  {} == {}'.format(Effect_token[-1], Text_token[Index_Effect_token_end[i]]))
                        print("")
                except:
                        error_sentence.append(i)             
                        Index_Cause_token_start[i] = (0)
                        Index_Cause_token_end[i] = (0)
                        Index_Effect_token_start[i] = (0)
                        Index_Effect_token_end[i] = (0)
        print("error_sentence",error_sentence)

        Index_df = pd.DataFrame(Index, columns=['Index'])
        Text_df = pd.DataFrame(Text, columns=['Index'])
        Cause_Start_df = pd.DataFrame(Index_Cause_token_start, columns=['Cause_Start'])
        Cause_End_def = pd.DataFrame(Index_Cause_token_end, columns=['Cause_End'])
        Effect_Start_df = pd.DataFrame(Index_Effect_token_start, columns=['Effect_Start'])
        Effect_End_def = pd.DataFrame(Index_Effect_token_end, columns=['Effect_End'])
        new_df = pd.concat([Index_df,Text_df,Cause_Start_df, Cause_End_def, Effect_Start_df,Effect_End_def], axis=1)
        new_df.to_csv("./2020NLP/data/converted_offset_train.csv")
        # print(len(Index))
        # print(len(Index_Cause_token_end))
        return (Index,Text,Index_Cause_token_start,Index_Cause_token_end,Index_Effect_token_start,Index_Effect_token_end,error_sentence)

# #########
# # print(len(Cause_Start))
# # print(len(Effect_Start))
# # for i, t in enumerate(Text):
# #         t = t + " "
# #         Text[i] = t

# Index_Cause_token_start = []
# Index_Cause_token_end = []
# Index_Effect_token_start = []
# Index_Effect_token_end = []
# for i in range(len(Text)):
#         print('####### {} th sentence #######'.format(i))
#         Text_token = tokenizer.tokenize(str(Text[i]))
#         print("Text_token:",Text_token)
#         print("len(Text_token:",len(Text_token))
#         Cause_token = tokenizer.tokenize(str(Cause[i]))
#         print("Cause_token",Cause_token)
#         print("len(Cause_token",len(Cause_token))
#         Effect_token = tokenizer.tokenize(str(Effect[i]))
#         print("Effect_token",Effect_token)
#         print("len(Effect_token",len(Effect_token))
#         # count =0
#         # no_cause_effect = []
#         # if Cause_token == "" and Effect_token == "":
#         #         Index_Cause_token_start.append(0)
#         #         Index_Cause_token_end.append(0)
#         #         Index_Effect_token_start.append(0)
#         #         Index_Effect_token_end.append(0)
#         #         count+=1
#         #         break
#         # if Cause_token == "":
#         #         Index_Cause_token_start.append(0)
#         #         Index_Cause_token_end.append(0)
#         # if Effect_token == "":
#         #         Index_Effect_token_start.append(0)
#         #         Index_Effect_token_end.append(0)
#         # else:
#         for Index in range(len(Text_token)):
#         # print(Cause_token[0], Text_token[Index])
#         # print(Cause_token[1], Text_token[Index+1])
#                 try:
#                         if Cause_token[0] == Text_token[Index] and Cause_token[1] == Text_token[Index+1] and Cause_token[2] == Text_token[Index+2] and Cause_token[3] == Text_token[Index+3] and Cause_token[4] == Text_token[Index+4]:
#                                 Index_Cause_token_start.append(Index)
#                                 Index_Cause_token_end.append(Index+len(Cause_token)-1)
#                 except:
#                         pass
#                         # if Cause_token[0] == Text_token[Index] and Cause_token[1] == Text_token[Index+1] and Cause_token[2] == Text_token[Index+2] and Cause_token[3] == Text_token[Index+3]:
#                         #         Index_Cause_token_start.append(Index)
#                         #         Index_Cause_token_end.append(Index+len(Cause_token)-1) 

#                         # if Cause_token[0] == Text_token[Index] and Cause_token[1] == Text_token[Index+1] and Cause_token[2] == Text_token[Index+2]:
#                         #         Index_Cause_token_start.append(Index)
#                         #         Index_Cause_token_end.append(Index+len(Cause_token)-1)  

#                 # try:                
#                 #         if Cause_token[-1] == Text_token[Index] and Cause_token[-2] == Text_token[Index-1] and Cause_token[-3] == Text_token[Index-2]:
#                 #                 Index_Cause_token_end.append(Index)
#                 # except:
#                 #         continue
#                 try:                            
#                         if Effect_token[0] == Text_token[Index] and Effect_token[1] == Text_token[Index+1] and Effect_token[2] == Text_token[Index+2] and Effect_token[3] == Text_token[Index+3]:
#                                 Index_Effect_token_start.append(Index)
#                                 Index_Effect_token_end.append(Index+len(Effect_token)-1)  
#                 except:
#                         try:
#                                 if Effect_token[0] == Text_token[Index] and Effect_token[1] == Text_token[Index+1] and Effect_token[2] == Text_token[Index+2]:
#                                         Index_Effect_token_start.append(Index)
#                                         Index_Effect_token_end.append(Index+len(Effect_token)-1)  
#                         except:
#                                 pass
#                 # try:
#                 #         if Effect_token[-1] == Text_token[Index] and Effect_token[-2] == Text_token[Index-1] and Effect_token[-3] == Text_token[Index-2]:
#                 #                 Index_Effect_token_end.append(Index)
#                 # except:
#                 #         continue   

                                                                                  
#         print('Text: ',Text[i])
#         print('Cause: ',Cause[i])
#         print('Effect: ',Effect[i])
#         print('Cause_token[0] == Text_token[Index]   :  {} == {}'.format(Cause_token[0], Text_token[Index_Cause_token_start[i]]))
#         print('Cause_token[-1] == Text_token[Index]  :  {} == {}'.format(Cause_token[-1], Text_token[Index_Cause_token_end[i]]))
#         print('Effect_token[0] == Text_token[Index]   :  {} == {}'.format(Effect_token[0], Text_token[Index_Effect_token_start[i]]))
#         print('Effect_token[-1] == Text_token[Index]  :  {} == {}'.format(Effect_token[-1], Text_token[Index_Effect_token_end[i]]))
#         print("")

# print("NO CAUSE or NO EFFECT COUNT: {}".format(count))

# Index_df = pd.DataFrame(Index, columns=['Index'])
# Text_df = pd.DataFrame(Text, columns=['Index'])
# Cause_Start_df = pd.DataFrame(Index_Cause_token_start, columns=['Cause_Start'])
# Cause_End_def = pd.DataFrame(Index_Cause_token_end, columns=['Cause_End'])
# Effect_Start_df = pd.DataFrame(Index_Effect_token_start, columns=['Effect_Start'])
# Effect_End_def = pd.DataFrame(Index_Effect_token_end, columns=['Effect_End'])
# new_df = pd.concat([Cause_Start_df, Cause_End_def, Effect_Start_df,Effect_End_def], axis=1)
# new_df.to_csv("./2020NLP/data/converted_offset_train.csv")

