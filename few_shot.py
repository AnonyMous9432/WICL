import pickle
import pandas as pd
import re
import random
from collections import OrderedDict
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle as pkl
import argparse
from math import ceil
import os
import shutil
from copy import deepcopy
from typing import List
import inspect
from collections import namedtuple
from csv import DictReader
import torch
import numpy
from vllm import LLM, SamplingParams
from transformers import pipeline,AutoTokenizer,AutoModel
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import huggingface_hub
hf_auth = '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' # Huggingface Authorization token is required which can be accessed upon request from this link https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
huggingface_hub.login(hf_auth)
llm = LLM(model="meta-llama/Llama-2-13b-chat-hf")


## for 2 class zero-shot
instrction_txt_zero2_1='''Your task as a scientific fact verifier is to analyze claims and determine their claim label, which can be either 'True' or 'False'.\n'''
instrction_txt_zero2_2='''Given a claim, you should provide a response in the format {"label": "class"}.'''

## for 3 class zero-shot
instrction_txt_zero3_1='''Your task as a scientific fact verifier is to analyze claims and determine their claim label, which can be either 'True', 'False' or 'Not Enough Information'.\n'''
instrction_txt_zero3_2='''Given a claim, you should provide a response in the format {"label": "class"}.'''

## for 2 class few-shot
instrction_txt_fw2_1='''Your task as a fact verifier is to analyze claims and determine their claim label, which can be either 'True' or 'False'.
Related context:\n'''
instrction_txt_fw2_2='''Given a claim, you should provide a response in the format {"label": "class"}.'''

## for 3 class few-shot
instrction_txt_fw3_1='''Your task as a fact verifier is to analyze claims and determine their claim label, which can be either 'True', 'False' or 'Not Enough Information'.'''
instrction_txt_fw3_2='''Thus given a claim, you should provide a response in the format {"label": "class"}.'''

## for 3 class classification
label_id={'Not Enough Information': 2, 'False': 0, 'True': 1,"Partially True":1,'true':1,'false':0, "Mostly False":0, "Mostly True":1}

## for 2 class classification
label_id={ 'False': 0, 'True': 1,"Partially True":1,'true':1,'false':0, "Mostly False":0, "Mostly True":1}

def get_dict_clswse_lbl(data,claim_id_claim,claim_id_labels, flag='three'):    # change the flag, according to 3_way/ 2 way
    clm_id=data['cl_id'].tolist()
    claim_lst=data['claim'].tolist()
    label_lst=data['label'].tolist()
    claim_id_claim=dict(zip(clm_id,claim_lst))   ## unchange this dictionary in case of 3 way classification
    claim_id_labels=dict(zip(clm_id,label_lst))  ## unchange this dictionary in case of 3 way classification
    if flag=='twoComb':
        gropher_2wy_clmid_lbl=OrderedDict()
        for j in claim_id_labels:
            if claim_id_labels[j]==2:
                gropher_2wy_clmid_lbl[j]=0
            else:
                gropher_2wy_clmid_lbl[j]=claim_id_labels[j]
#         gropher_2wy_clm_id_clms=deepcopy(claim_id_claim)
        new_set_clm_id_clms=deepcopy(claim_id_claim)
        new_set_clmid_lbl=deepcopy(gropher_2wy_clmid_lbl)
    elif flag=='two':
        ## in case of 2 way classification, i.e., SUPPORT and REFUTE class, ignore NEI.
        non_grophr_2wy_clmid_lbl=deepcopy(claim_id_labels)
        non_grophr_2wy_clmid_clm=deepcopy(claim_id_claim)
        for i in claim_id_labels:
            if claim_id_labels[i]==2:
                non_grophr_2wy_clmid_lbl.pop(i)
                non_grophr_2wy_clmid_clm.pop(i)
        new_set_clm_id_clms=deepcopy(non_grophr_2wy_clmid_clm)
        new_set_clmid_lbl=deepcopy(non_grophr_2wy_clmid_lbl)
    else:
        new_set_clm_id_clms=deepcopy(claim_id_claim)
        new_set_clmid_lbl=deepcopy(claim_id_labels)
    return new_set_clm_id_clms, new_set_clmid_lbl


def get_zero_shot_fw_shot(data,data_wiki,claim_id_claim, flag='orc',num_s=1):    # change the flag, according to zeroshot, fewshot or oracle setting
    prompt_list=OrderedDict()
    if flag=='few':
        for i in data:
            prompt_txt="{}\n".format(instrction_txt_fw3_1)  #change
            top_docs=[]
            label_top=[]
            for j in data_wiki[i]:
                top_docs.append(data_wiki[i][j][0].strip('\n'))
                label_top.append(data_wiki[i][j][2])
            prompt_txt=prompt_txt+"Given that:"
            for l,k in enumerate(top_docs[:num_s-1]): #for 3shot, k=2
                label_=label_top[l]
                evidence=k.strip('\n')
                prompt_txt=prompt_txt+''' {} is {},'''.format(evidence,label_)
            prompt_txt=prompt_txt+''' {} is {}.'''.format(top_docs[num_s].strip('\n'),label_top[2])
#             prompt_txt=prompt_txt+'''and {} is {}.'''.format(top_docs[num_s].strip('\n'),label_top[2])
            prompt_txt=prompt_txt+"\n"+instrction_txt_fw3_2
            prompt_n=prompt_txt+"\nInput: {}\nOutput: ".format(data[i])
            prompt_list[i]=prompt_n
    elif flag=='zero':
        for i in data:
            prompt_txt="{}\n".format(instrction_txt_zero2_1)
            prompt_txt=prompt_txt+instrction_txt_zero2_2
            prompt_n=prompt_txt+"\nInput: {}\nOutput: ".format(claim_id_claim[i])
            prompt_list[i]=prompt_n
    else:
        for i in data:
            prompt_txt="{}".format(instrction_txt_fw2_1)
            top_docs=data_wiki[i]['evidence'] ## need to change accordingly
            #print(top_docs)
            for k in top_docs[:num_s]:
                evidence=k.strip('\n')
                prompt_txt=prompt_txt+'''Input: {}\n'''.format(evidence)
            prompt_txt=prompt_txt+instrction_txt_fw2_2
            prompt_n=prompt_txt+"\nInput: {}\nOutput: ".format(claim_id_claim[i])
            prompt_list[i]=prompt_n
    return prompt_list

def isWordPresent(sentence, word):
    s = sentence.split("\n")
    if word in sentence:
        return True
    return False

def icl_output_analysis(data,flag=2):  ## change the flag accoridng to 2 class/ 3 class classification task
    extrcted_lst=[]
    not_parsed=[]
    dict_val={}
    for i in tqdm(data):
        try:
            start=data[i].outputs[0].text.strip().find('{')
            end=data[i].outputs[0].text.strip().find('}')
            json_txt=data[i].outputs[0].text.strip()[start:end+1]
            try:
                extrcted_lst.append(label_id[str(ast.literal_eval(json_txt)['label'])])
                dict_val[i]=label_id[str(ast.literal_eval(json_txt)['label'])]
            except:
                extrcted_lst.append(label_id[ast.literal_eval(json_txt)['label']])
                dict_val[i]=label_id[str(ast.literal_eval(json_txt)['label'])]
        except:
            json_txt=data[i].outputs[0].text.strip()
            a=isWordPresent(json_txt, "True")
            b=isWordPresent(json_txt, "False")
            if a==True:
                dict_val[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif b==True:
                dict_val[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif flag==2:
                not_parsed.append(data[i])
                extrcted_lst.append(0)
                dict_val[i]=data[i].outputs[0].text.strip()
            elif flag==3:
                not_parsed.append(data[i])
                extrcted_lst.append(2)
                dict_val[i]=data[i].outputs[0].text.strip()
    return extrcted_lst,dict_val,not_parsed

def get_true_labels(data,lbls):
    true_labels=[]
    count=0
    for i in data:
        lab=lbls[i]
        true_labels.append(lab)
    return true_labels


def main(args):
    DATASET_PATH = args.dataset_path
    CORPUS_PATH = args.corpus_path
    SAMPLES = args.n_smpls
    FLAG1= args.flag1
    FLAG2= args.flag2
    FLAG3 =args.flag3


#     #load template
#     texts_nw=pkl.load(open(TEMP_PATH,"rb"))
#     dataframe=make_result_file(dataset_path=DATASET_PATH,output_path_result=RESULT_PATH,output_path_probability=PROB_PATH,classification_report_file_path=REPORT_FILE_PATH,variable=VAR,texts=texts_nw,out_path=PRED_FILE_PATH)

    # #input your data(3 class) in csv format
    data1=pd.read_csv(DATASET_PATH)

    clm_id=data1['cl_id'].tolist()
    claim_lst=data1['claim'].tolist()
    label_lst=data1['label'].tolist()
    claim_id_claim=dict(zip(clm_id,claim_lst))   ## unchange this dictionary in case of 3 way classification
    claim_id_labels=dict(zip(clm_id,label_lst))  ## unchange this dictionary in case of 3 way classification

    ## top retrieved sentences from wikipedia corpus (FEVER dataset)/ CORD19 corpus (SciFact and Check Covid dataset)
    data_wiki=pkl.load(open(CORPUS_PATH,"rb"))

    sampling_params = SamplingParams(temperature=0.0, max_tokens=10,top_p=1,length_penalty=0.7, best_of=10,use_beam_search=True,early_stopping=True)

    non_grophr_2wy_clmid_clm, non_grophr_2wy_clmid_lbl=get_dict_clswse_lbl(data=data1,claim_id_claim=claim_id_claim, claim_id_labels=claim_id_labels, flag=FLAG2)
    non_grophr_2wy_clm_0_ls=get_zero_shot_fw_shot(data=non_grophr_2wy_clmid_clm, data_wiki=data_wiki, flag=FLAG2, claim_id_claim=claim_id_claim, num_s= SAMPLES )
    # non_grophr_2wy_clm_1_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,FLAG2,SAMPLES)   #few shot, FLAG2='few', SAMPLES=1
    # non_grophr_2wy_clm_2_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,'few',2)
    # non_grophr_2wy_clm_3_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,'few',3)
    # non_grophr_2wy_clm_4_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,'few',4)
    # non_grophr_2wy_clm_5_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,'few',5)
    # non_grophr_2wy_clm_6_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,'few',6)
    # non_grophr_2wy_clm_7_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,'few',7)
    # non_grophr_2wy_clm_orc_ls=get_zero_shot_fw_shot(non_grophr_2wy_clmid_clm,num_s=2)

    # all_3wy_clm_1_ls=get_zero_shot_fw_shot(claim_id_claim,'few',4)    ## 3 class classification, 4shot
#     non_grophr_2wy_clm_1_ls=get_zero_shot_fw_shot(gropher_2wy_clm_id_clms,'few',5)    ## non-gopher style 2 class classification, 5shot
    responses = llm.generate(list(non_grophr_2wy_clm_0_ls.values()), sampling_params)
    response_dict=dict(zip(list(non_grophr_2wy_clm_0_ls.keys()),responses))
    pred_out,pred_dict,a=icl_output_analysis(response_dict,flag=FLAG3)
    true_labels_req=get_true_labels(pred_dict,non_grophr_2wy_clmid_lbl)

    pred=classification_report(true_labels_req,pred_out,digits=4)
    f1_score=f1_score(true_labels_req,pred_out, average='micro')
    accuracy_score=accuracy_score(true_labels_req,pred_out)

    report = classification_report(true_labels_req,pred_out, output_dict=True)
    classwise_metrics = {}
    for label, metrics in report.items():
        if label.isdigit():
            classwise_metrics[int(label)] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score']
            }
    for label, metrics in classwise_metrics.items():
        print(f"Class {label}:")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall: {metrics['recall']}")
        print(f"  F1-Score: {metrics['f1-score']}")


if __name__ == "__main__":
  parser= argparse.ArgumentParser(description="WICL")

  parser.add_argument("--dataset_path",type=str,default='/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/splitted_data/1_shot.csv',
                      help="Give the dataset path .csv format properly")

  parser.add_argument("--corpus_path",
                      type=str,default='/media/pbclab/77a4ead7-6f71-4061-ac44-73526c7285f2/payel/data_open_prompt/data/splitted_data/knn/top_50_bert_vec_test_claim_new', help="Give the  path for corpus")
  parser.add_argument("--flag1",
                        type=str,default='two', help="Give three/two/twoComb, for selecting Combined 2way/ 2way/ 3 way")
  parser.add_argument("--flag2",
                          type=str,default='zero', help="Give zero/few/orc, for zeroshot #n_smpls is not required, but for fewshot #n_smpls should be defined accordingly")
  parser.add_argument("--flag3",
                          type=int,default=2, help="Provide 2/3, for classwise output analysis")


  parser.add_argument("--n_smpls",
                      type=int,
                      default = 1,
                      help="Give 1,2,3,4,5,6 according to shots",
                      )

#   parser.add_argument("--batch_size",type=int,
#                       default = 128,
#                       help = "Number of data per batch. Default is 16")



  args = parser.parse_args()

  print(args)
  main(args)
