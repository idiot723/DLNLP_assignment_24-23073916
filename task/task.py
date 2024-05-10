import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments,  
                          logging)
from nltk.translate.bleu_score import sentence_bleu
import logging
from bleurt import score
from bert_score import score as score1
logging.disable(logging.INFO)
from pathlib import Path

def install_packages():
    subprocess.call(["pip", "install", "-q", "-U", "torch", "--index-url", "https://download.pytorch.org/whl/cu117"])
    subprocess.call(["pip", "install", "-q", "-U", "-i", "https://pypi.org/simple/", "bitsandbytes"])
    subprocess.call(["pip", "install", "-q", "-U", "transformers==4.40.0"])
    subprocess.call(["pip", "install", "-q", "-U", "accelerate"])
    subprocess.call(["pip", "install", "-q", "-U", "datasets"])
    subprocess.call(["pip", "install", "-q", "-U", "trl"])
    subprocess.call(["pip", "install", "-q", "-U", "peft"])
    subprocess.call(["pip", "install", "-q", "-U", "tensorboard"])
    subprocess.call(["pip", "install", "-q", "-U", "einops"])

def load_data():
  
    train_df = pd.read_csv("./Dataset/train_df.csv")
    test_df = pd.read_csv("./Dataset/test_df.csv")

    return train_df, test_df

def generate_prompt(data_point):
    return f"""请将以下文言文句子翻译为现代汉语：

    文言文句子：{data_point["content"]}
    
    翻译结果：{data_point["translation"]}"""

def generate_test_prompt(data_point):
    return f"""请将以下文言文句子翻译为现代汉语：

    文言文句子：{data_point["content"].strip("'")}
    
    翻译结果：
    """
def predict(test, model, tokenizer):
    y_pred = []
    for prompt in tqdm(test["translations"]):
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(input_ids=inputs.input_ids, 
                         max_length = 230,  
                         temperature=0.0,  
                         num_return_sequences=1,  
                         pad_token_id=tokenizer.pad_token_id,  
                         eos_token_id=tokenizer.eos_token_id,  
                         )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        answer = generated_text.split("翻译结果：")[-1].strip()
        y_pred.append(answer)
    return y_pred

def load_model(model_name):
    # model_name = "microsoft/Phi-3-mini-4k-instruct" 

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=False,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      trust_remote_code=True,
      device_map="auto",
      quantization_config=bnb_config, 
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    max_seq_length = 2048
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct" , 
                                          trust_remote_code=True,
                                          max_seq_length=max_seq_length,
                                         )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def model_tuning(model,train_data,valid_data,tokenizer):
    peft_config = LoraConfig(
      r=16,
      lora_alpha=16,
      target_modules="all-linear",
      lora_dropout=0.00,
      bias="none",
      task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
      output_dir="logs",
      num_train_epochs=4,
      per_device_train_batch_size=1,
      gradient_accumulation_steps=8, # 4
      optim="paged_adamw_32bit",
      save_steps=0,
      logging_steps=25,
      learning_rate=5e-5,
      weight_decay=0.001,
      fp16=True,
      bf16=False,
      max_grad_norm=0.3,
      max_steps=-1,
      warmup_ratio=0.03,
      group_by_length=True,
      lr_scheduler_type="cosine",
      report_to="tensorboard",
      evaluation_strategy="epoch"
    )
    max_seq_length = 2048
    trainer = SFTTrainer(
      model=model,
      train_dataset=train_data,
      eval_dataset=valid_data,
      peft_config=peft_config,
      dataset_text_field="translations",
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      args=training_arguments,
      packing=False,
    )
    #start train
    trainer.train()
    # Save trained model
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    trainer.model.save_pretrained(parent_dir /"trained-model3")#"trained-model2"/"trained-model"

def calculate_scores(df):
    bleu_scores = []
    bleurt_scores = []
    bert_scores = []

    checkpoint = "./task/BLEURT-20/BLEURT-20"
    scorer = score.BleurtScorer(checkpoint)


    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # get reference text and generated text
        reference = str(row['translation'])  
        translation = str(row['modern_translation'])  

        # calculate bleu score
        bleu_score = sentence_bleu([reference.split()], translation.split())
        bleu_scores.append(bleu_score)

        # # calculate bleurt score
        scores = scorer.score(references=[reference], candidates=[translation])
        bleurt_scores.append(scores)

        # calculate BERTScore 
        _, _, bert_score = score1([translation], [reference], lang="zh", verbose=False)
        bert_scores.append(bert_score.mean().item())


    # add scores into DataFrame
    df['bleu_score'] = bleu_scores
    df['bleurt_score'] = bleurt_scores
    df['bert_score'] = bert_scores

    return df

def get_mean(df):
    df['bleurt_score'] = df['bleurt_score'].apply(lambda x: x[1:-1] if isinstance(x, str) else x)  # 去除列表的括号
    df['bleurt_score'] = df['bleurt_score'].apply(lambda x: float(x[0]) if isinstance(x, list) else float(x))  # 将列表中的第一个元素转换为浮点数
    avg_score = df['bleu_score'].mean()
    avg_score2 = df['bleurt_score'].mean()
    avg_score3 = df['bert_score'].mean()
    print([avg_score,avg_score2,avg_score3])
    return df

def trim_translation(row):
    content = row['translation']
    modern_translation = row['modern_translation']
    sentences = content.split('，')
    trimmed_translation = '，'.join(modern_translation.split('，')[:len(sentences)+1])
    return trimmed_translation

def calcu_scores(train_df,test_sdf):
    #read data
    df1 = pd.read_csv('./task/origin_test.csv')
    df2 = pd.read_csv('./task/origin_train.csv')
    df3 = pd.read_csv('./task/800_testmodel.csv')
    df4 = pd.read_csv('./task/800_trainmodel.csv')
    df5 = pd.read_csv('./task/2000_testmodel.csv')
    df6 = pd.read_csv('./task/2000_model.csv')
    train_val = train_df.head(500)

    #data process
    df1['modern_translation'] = df1['modern_translation'].str.split('\n').str[0]
    df2['modern_translation'] = df2['modern_translation'].str.split('\n').str[0]
    df3['modern_translation'] = df3['modern_translation'].str.split('\n').str[0]
    df4['modern_translation'] = df4['modern_translation'].str.split('\n').str[0]
    df5['modern_translation'] = df5['modern_translation'].str.split('\n').str[0]
    df5['modern_translation'] = df5['modern_translation'].astype(str) 
    df5['modern_translation'] = df5.apply(trim_translation, axis=1) 
    df6['modern_translation'] = df6['modern_translation'].str.split('\n').str[0]

    new_df1 = pd.concat([test_sdf['translation'], df1['modern_translation']], axis=1)
    new_df3 = pd.concat([test_sdf['translation'], df3['modern_translation']], axis=1)
    new_df5 = pd.concat([test_sdf['translation'], df5['modern_translation']], axis=1)

    new_df2 = pd.concat([train_val['translation'], df2['modern_translation']], axis=1)
    new_df4 = pd.concat([train_val['translation'], df4['modern_translation']], axis=1)
    new_df6 = pd.concat([train_val['translation'], df6['modern_translation']], axis=1)


    new_df1.fillna(value="", inplace=True)
    new_df2.fillna(value="", inplace=True)
    new_df3.fillna(value="", inplace=True)
    new_df4.fillna(value="", inplace=True)
    new_df5.fillna(value="", inplace=True)
    new_df6.fillna(value="", inplace=True)

    new_df1 = calculate_scores(new_df1) #pd.read_csv("./task/origin_ontest.csv")
    print("original phi-3 model scores on test dataset")
    new_df1 = get_mean(new_df1)
    new_df1.to_csv("./task/origin_ontest.csv", index=False)

    new_df2 = pd.read_csv("./task/origin_ontrain.csv")#calculate_scores(new_df2)
    print("original phi-3 model scores on train dataset")
    new_df2 = get_mean(new_df2)
    new_df2.to_csv("./task/origin_ontrain.csv", index=False)

    new_df3 = pd.read_csv("./task/800_ontest.csv")#calculate_scores(new_df3)
    print("phi-3 tuned with 800 data model scores on test dataset")
    new_df3 = get_mean(new_df3)
    new_df3.to_csv("./task/800_ontest.csv", index=False)

    new_df4 = pd.read_csv("./task/800_ontrain.csv")#calculate_scores(new_df4)
    print("phi-3 tuned with 800 data model scores on train dataset")
    new_df4 = get_mean(new_df4)
    new_df4.to_csv("./task/800_ontrain.csv", index=False)

    new_df5 = pd.read_csv("./task/2000_ontest.csv")#calculate_scores(new_df5)
    print("phi-3 tuned with 2000 data model scores on test dataset")
    new_df5 = get_mean(new_df5)
    new_df5.to_csv("./task/2000_ontest.csv", index=False)

    new_df6 = pd.read_csv("./task/2000_ontrain.csv")#calculate_scores(new_df6)
    print("phi-3 tuned with 2000 data model scores on train dataset")
    new_df6 = get_mean(new_df6)
    new_df6.to_csv("./task/2000_ontrain.csv", index=False)

def new_prompt_train_score(train_df):
    df7  = pd.read_csv("./task/800_change_prompt_train.csv")
    train_sval = train_df.head(100)
    df7['modern_translation'] = df7['modern_translation'].str.split('\n').str[0]
    new_df7 = pd.concat([train_sval['translation'], df7['modern_translation']], axis=1)
    new_df7.fillna(value="", inplace=True)
    new_df7 = calculate_scores(new_df7)
    new_df7 = get_mean(new_df7)
    new_df7.to_csv('./task/800_prompt_train.csv', index=False)

def new_prompt_test_score(test_df):
    df8  = pd.read_csv("800_change_prompt.csv")
    test_sdf = test_df.head(100)
    df8['modern_translation'] = df8['modern_translation'].str.split('\n').str[0]
    new_df8 = pd.concat([test_sdf['translation'], df8['modern_translation']], axis=1)
    new_df8.fillna(value="", inplace=True)
    new_df8.head()
    new_df8 = calculate_scores(new_df8)
    new_df8 = get_mean(new_df8)
    new_df8.to_csv('./task/800_prompt_test.csv', index=False)

def plot_histograms(df1, df2, df3, score, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].hist(df1[score], bins=20, color='skyblue', edgecolor='black')
    axs[0].set_xlabel(score)
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Distribution of {score} (origin)')

    axs[1].hist(df2[score], bins=20, color='skyblue', edgecolor='black')
    axs[1].set_xlabel(score)
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Distribution of {score} (with 800)')

    axs[2].hist(df3[score], bins=20, color='skyblue', edgecolor='black')
    axs[2].set_xlabel(score)
    axs[2].set_ylabel('Frequency')
    axs[2].set_title(f'Distribution of {score} (with 2000)')

    plt.tight_layout()
    plt.savefig(path)
    #plt.show()

def plot_test(score,path):
    new_df1 = pd.read_csv("./task/origin_ontest.csv")
    new_df3 = pd.read_csv("./task/800_ontest.csv")
    new_df5 = pd.read_csv("./task/2000_ontest.csv")
    plot_histograms(new_df1, new_df3, new_df5, score,path)

def plot_train(score,path):
    new_df2 = pd.read_csv("./task/origin_ontrain.csv")
    new_df4 = pd.read_csv("./task/800_ontrain.csv")
    new_df6 = pd.read_csv("./task/2000_ontrain.csv")
    plot_histograms(new_df2, new_df4, new_df6, score,path)

