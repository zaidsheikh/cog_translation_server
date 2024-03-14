"""
python train.py --train --in_train=data/lao_Laoo.dev --out_train=data/eng_Latn.dev --in_test=data/lao_Laoo.devtest --out_test=data/eng_Latn.devtest --in_lang=lao_Laoo --out_lang=eng_Latn --model=facebook/nllb-200-distilled-600M
python train.py --in_test=data/tweets.en-my.my --in_lang=mya_Mymr --out_lang=eng_Latn --model=./checkpoint-lo/checkpoint-7000
"""

import os
import sys
import argparse
from tqdm import tqdm
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
from datasets import Dataset, concatenate_datasets
import numpy as np
import gc
import torch
from torch.utils.checkpoint import checkpoint
import evaluate

metric = evaluate.load("sacrebleu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--in_train', type=str, help = 'Training file: Input')
    parser.add_argument('--out_train', type=str, help = 'Training file: Output')
    parser.add_argument('--in_test', type=str, help = 'Testing file: Input')
    parser.add_argument('--out_test', type=str, help = 'Testing file: Output')
    parser.add_argument('--in_lang', type=str, help = 'Input Language (for tokenizer)')
    parser.add_argument('--out_lang', type=str, help = 'Output Language (for tokenizer)')
    parser.add_argument('--model', type=str, help = 'Model name (for model instantiation)')
    parser.add_argument('--model_tag', type=str, help = 'Model name (for resulting file)')
    parser.add_argument('--mult_splits', action='store_true')
    return parser.parse_args()

def load_data(in_file, out_file, sp, args):
    data = []
    tokenizer = NllbTokenizer.from_pretrained(args.model, src_lang=args.in_lang, tgt_lang=args.out_lang)
    if args.mult_splits:
        in_file = os.path.splitext(in_file)[0] + "." + str(sp) + os.path.splitext(in_file)[1]
        out_file = os.path.splitext(out_file)[0] + "." + str(sp) + os.path.splitext(out_file)[1]
    with open(in_file) as fi, open(out_file) as fo:
       data = Dataset.from_dict({
           "src": [i.strip() for i in fi],
           "tgt": [i.strip() for i in fo]
           })
    return data.map(lambda x: tokenizer(text=x["src"], text_target=x["tgt"], padding='max_length', truncation=True, max_length=128))

def evaluate(model, tokenizer, args):
    # translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=args.in_lang, tgt_lang=args.out_lang, batch_size=1, device_map="auto")
    in_file = args.in_test
    if args.mult_splits:
        in_file = os.path.splitext(in_file)[0] + "." + str(sp) + os.path.splitext(in_file)[1]
    # num_lines = sum(1 for line in open(in_file,'r'))
    out = []
    with open(in_file, "r") as f:
        lines = f.read().splitlines()
    batch_size=256
    for i in tqdm(range(0, len(lines), batch_size)):
        batch = lines[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs) #, num_beams=5, num_return_sequences=5) # use generate() instead of pipelne() for sampling strategies
        out += tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return out

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    print("decoded_preds", decoded_preds[0])
    print("decoded_labels", decoded_labels[0])

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    print("result", result)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


if __name__ == "__main__":
    args = parse_args()
    num_splits = 3 if args.mult_splits else 1
    for sp in range(num_splits):
        if args.train:
            print("Loading dataset...")
            ## MODIFY ADDIITIONAL DATA HERE ##
            train_data = concatenate_datasets([load_data(args.in_train, args.out_train, sp, args), load_data('./data/train.en-lo.lo','./data/train.en-lo.en', sp, args)]).shuffle(seed=42)
            eval_data = load_data(args.in_test, args.out_test, sp, args)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to("cuda")
            ## MODIFY THE CHECKPOINT DIR HERE ##
            training_args = Seq2SeqTrainingArguments(output_dir="./checkpoint-lo",
                                                    save_strategy="steps",
                                                    seed=42,
                                                    gradient_accumulation_steps=8,
                                                    evaluation_strategy="steps",
                                                    learning_rate=2e-5,
                                                    # per_device_train_batch_size=8,
                                                    auto_find_batch_size=True,
                                                    per_device_eval_batch_size=1,
                                                    weight_decay=0.01,
                                                    save_total_limit=5,
                                                    num_train_epochs=15,
                                                    predict_with_generate=True,
                                                    fp16=True,
                                                    metric_for_best_model='eval_loss',
                                                    load_best_model_at_end=True,
                                                )
            tokenizer = NllbTokenizer.from_pretrained(args.model, src_lang=args.in_lang, tgt_lang=args.out_lang) # needed for compute_metrics
            trainer = Seq2SeqTrainer(model, training_args, train_dataset=train_data, eval_dataset=eval_data, compute_metrics=compute_metrics)
            gc.collect()
            torch.cuda.empty_cache()
            print("Fine-tuning model...")
            ### RESUME FROM CHECKPOINT HERE ##
            trainer.train(resume_from_checkpoint="./checkpoint-lo/checkpoint-5500")

            print("Evaluating trained model...")
            out = evaluate(model, tokenizer, args)
            output_file = "./outputs/last.en"
            with open(output_file, "w") as f:
                f.write("\n".join([str(o) for o in out]))
        else:
            tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', src_lang=args.in_lang, tgt_lang=args.out_lang)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to("cuda")
            out = evaluate(model, tokenizer, args)
            output_file = "./outputs/checkpoint.en" 
            with open(output_file, "wb") as f:
                f.write("\n".join([o for o in out]).encode('utf-8').strip())

