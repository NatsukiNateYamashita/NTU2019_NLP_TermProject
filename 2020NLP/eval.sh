#!/bin/sh
var=0
while [ $var -lt 100 ]  #範囲の書き方(Bash独自) => {0..4}
do
        python3 /nfs/nas-7.1/yamashita/NTU_NLP/task2/2020NLP/task2_evaluate.py from-file --ref_file /nfs/nas-7.1/yamashita/NTU_NLP/task2/2020NLP/data/train.csv /nfs/nas-7.1/yamashita/NTU_NLP/task2/csv_save/bert-base-uncased_b_16_e_${var}.csv /nfs/nas-7.1/yamashita/NTU_NLP/task2/csv_save/output_bert-base-uncased_b_16_e_${var}.csv
        var=`expr $var + 1`
done
var=0
while [ $var -lt 100 ]  #範囲の書き方(Bash独自) => {0..4}
do
        python3 /nfs/nas-7.1/yamashita/NTU_NLP/task2/2020NLP/task2_evaluate.py from-file --ref_file /nfs/nas-7.1/yamashita/NTU_NLP/task2/2020NLP/data/train.csv /nfs/nas-7.1/yamashita/NTU_NLP/task2/csv_save/bert-base-cased_b_16_e_${var}.csv /nfs/nas-7.1/yamashita/NTU_NLP/task2/csv_save/output_bert-base-cased_b_16_e_${var}.csv
        var=`expr $var + 1`
done