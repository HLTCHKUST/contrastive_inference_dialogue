import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.bleu.bleu import Bleu

from src.utils.cicero_prompt import SUBSEQ_EVENT_TEMPLATE, CAUSE_TEMPLATE, PREREQUISITE_TEMPLATE, REACTION_TEMPLATE, MOTIVATION_TEMPLATE
from src.utils.retriever import TfidfRetriever, normalize


def load_templates():
    return {
        "subseq_event": SUBSEQ_EVENT_TEMPLATE,
        "subseq_event_clipped": SUBSEQ_EVENT_TEMPLATE,
        "cause": CAUSE_TEMPLATE,
        "prerequisite": PREREQUISITE_TEMPLATE,
        "reaction": REACTION_TEMPLATE,
        "motivation": MOTIVATION_TEMPLATE,
        }

def get_question(relation):
    r2q = {
        "cause" : "What is or could be the cause of target?",
        "prerequisite" : "What is or could be the prerequisite of target?",
        "reaction" : "What is the possible emotional reaction of the listener in response to target?",
        "motivation" : "What is or could be the motivation of target?",
        "subseq_event" : "What subsequent event happens or could happen following the target?",
        "subseq_event_clipped" : "What subsequent event happens or could happen following the target?",
        }
    return r2q[relation]

def load_retriever(path):
    # load retriever
    tfidf_cause_retriever = TfidfRetriever(f"{path}/cause/train_cause_tfidf.tsv", f"{path}/cause/train_cause_tfidf-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    tfidf_motivation_retriever = TfidfRetriever(f"{path}/motivation/train_motivation_tfidf.tsv", f"{path}/motivation/train_motivation_tfidf-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    tfidf_prerequisite_retriever = TfidfRetriever(f"{path}/prerequisite/train_prerequisite_tfidf.tsv", f"{path}/prerequisite/train_prerequisite_tfidf-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    tfidf_reaction_retriever = TfidfRetriever(f"{path}/reaction/train_reaction_tfidf.tsv", f"{path}/reaction/train_reaction_tfidf-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    tfidf_subseq_event_retriever = TfidfRetriever(f"{path}/subseq_event/train_subseq_event_tfidf.tsv", f"{path}/subseq_event/train_subseq_event_tfidf-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    tfidf_subseq_event_clipped_retriever = TfidfRetriever(f"{path}/subseq_event_clipped/train_subseq_event_clipped_tfidf.tsv", f"{path}/subseq_event_clipped/train_subseq_event_clipped_tfidf-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")

    Domain2Retriever = {
        "cause": tfidf_cause_retriever,
        "motivation": tfidf_motivation_retriever,
        "prerequisite": tfidf_prerequisite_retriever,
        "reaction": tfidf_reaction_retriever,
        "subseq_event": tfidf_subseq_event_retriever,
        "subseq_event_clipped": tfidf_subseq_event_clipped_retriever,
    }
    return Domain2Retriever

def make_template(samples, d, t, r):
    prompt = []
    for sample in samples:
        sample_template = "Context:\n"
        question = get_question(sample[1])
        dialogues, target, answer = sample[0].split("\n")[0], sample[0].split("\n")[1], sample[0].split("\n")[2]
        dialogues = dialogues.replace(" <", "\n<")

        sample_template += dialogues + "\n"
        sample_template += "\nTarget:\n" + target
        sample_template += "\nQuestion:\n" + question
        sample_template += "\nAnswer:\n" + answer
        prompt.append(sample_template)
    
    q = get_question(r)
    sample_template = "Context:\n"
    sample_template += d + "\n"
    sample_template += "\nTarget:\n" + t
    sample_template += "\nQuestion:\n" + q
    sample_template += "\nAnswer:\n"
    prompt.append(sample_template)
    return "\n\n\n".join(prompt)


def filter_exemplar_context(samples, topk=5):
    new_samples = []
    record = {}
    for s in samples:
        ss = s[0].split("\n")
        assert len(ss) == 3
        if ss[0] in record:
            continue
        else:
            record[ss[0]] = True 
            new_samples.append(s)

            if len(new_samples) == topk:
                break
    return new_samples


def compute_metrics(scorers, predictions, golds):
    refs, hyps = {}, {}
    task_scores = {}
    for j in range(len(golds)):
        refs[j] = [golds[j]] if isinstance(golds[j], str) else golds[j]
        hyps[j] = [predictions[j]]

    for scorer, method in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(score, list):
            for m, s in zip(method, score):
                task_scores[m] = round(s, 5)
        else:
            task_scores[method] = round(score, 5)
    return task_scores


def main(args):
    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0")
    model.to(dtype=torch.float16, device=device)
    # load datasets
    data = load_dataset('src/data_utils/cicero.py', 'cicero_nlg')['test']
    # load retrievers
    retrievers = load_retriever(args.retriever_path)

    stop_token = tokenizer.eos_token

    core_filename = f"{args.save_path}/test"
    golds_filename = core_filename + "_gold.txt"
    generations_filename = core_filename + "_generation.txt"
    scores_filename = core_filename + "_scores.json"

    generated_sequences, gold_sequences = [], []
    for idx, row in enumerate(tqdm(data)):
        dialogue = row["dialogue"].replace(" <", "\n<")

        relation = row["relation"]
        query = normalize(relation) + " " + row["dialogue"] + "\n" + row["target"]
        exemplar_sample = retrievers[relation].get_KB(query, topk=args.filterk)
        exemplar_sample = filter_exemplar_context(exemplar_sample, topk=args.topk)

        retrieval_prompt = make_template(exemplar_sample, dialogue, row["target"], relation)

        # tokenize
        input_ids = tokenizer(retrieval_prompt, return_tensors='pt')
        input_gen_len = input_ids["input_ids"].shape[1]
        # generate
        generation = model.generate(
            input_ids=input_ids['input_ids'].to(device),
            attention_mask=input_ids['attention_mask'].to(device),
            pad_token_id=tokenizer.eos_token_id,
            max_length=input_gen_len+args.max_gen_len)
        text = tokenizer.decode(generation[0, input_gen_len:], skip_special_tokens=True) 
        text = text[: text.find(stop_token) if stop_token and text.find(stop_token)>0 else None]
        text = text[: text.find('\n')]
        generated_sequences.append(text)
        gold_sequences.append(row["answer"])
        if args.debug:
            print(f"The generated sentence is: {text}")
            print("The golden sentence is:", row["answer"])
            print("="*80)
            input()
            
    with open(generations_filename, "w") as f:
        for line in generated_sequences:
            f.write(line.replace("\n", " ")+"\n")
    with open(golds_filename, "w") as f:
        for line in gold_sequences:
            f.write(line.replace("\n", " ")+"\n")

    # Let's compute scores!
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    scores = compute_metrics(scorers, generated_sequences, gold_sequences)
    scores['sample_size'] = len(data)

    keys, values = [], []
    for k,v in scores.items():
        keys.append(k)
        values.append(str(round(v*100,2)))
    print(" ".join(keys))
    print(" ".join(values))
        
    with open(scores_filename, "w") as f:
        json.dump(scores, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enter DEBUG mode")
    parser.add_argument("--save_path", type=str, help="where to save generations", required=True)
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/gpt-j-6B", required=False)
    parser.add_argument("--max_gen_len", type=int, default=50, required=False)

    parser.add_argument("--retriever_path", type=str, help="where to load the retrievers", default="save/tfidf/cicero")
    parser.add_argument("--filterk", type=int, default=50, required=False)
    parser.add_argument("--topk", type=int, default=2, required=False)
    args = parser.parse_args()

    main(args)
