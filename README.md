# WICL
This repository contains source code for Claim Verification through Weakly Supervised In-context Learning.




## Inferencing
Run the following script to get the results for few-shot/zero-shot/oracle settings for 2way combined/2 way/3 way of FEVER/ SciFact/ Check Covid dataset
```bash
python3 few_shot.py --dataset_path='../data/1_shot.csv' \
--corpus_path='../data/top_40_wiki18_retrieved_3_class_fever.pickle' \
--flag1='two' \
--flag2='few'\
--flag3=2 \
--n_smpls=1
```
