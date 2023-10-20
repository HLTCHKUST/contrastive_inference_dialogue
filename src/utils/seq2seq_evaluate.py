import argparse
import json
import re
from collections import defaultdict

from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.bleu.bleu import Bleu


CICERO_DOMAIN = ["cause", "prerequisite", "reaction", "motivation", "subseq_event", "subseq_event_clipped"]
GLUCOSE_DOMAIN = ["cause", "motivation", "prerequisite_location", "prerequisite_posession", "prerequisite_other", "subseq_event", "reaction", "subseq_event_location", "subseq_event_posession", "subseq_event_other"]

class Evaluator():
    def __init__(self, args):
        self.data_file = args.data_file
        self.pred_file = args.pred_file
        self.out_dir = args.out_dir
        self.dataset_name = args.dataset
        self.domains = CICERO_DOMAIN if args.dataset == "cicero" else GLUCOSE_DOMAIN

    def _get_domain_cicero(self, q):
        questions = {
            "What is or could be the cause of target?" : "cause",
            "What is or could be the prerequisite of target?" : "prerequisite",
            "What is the possible emotional reaction of the listener in response to target?" : "reaction",
            "What is or could be the motivation of target?" : "motivation",
            "What subsequent event happens or could happen following the target?" : "subseq_event",
            }
        return questions[q]

    def _get_domain_glucose(self, dimension):
        domain_mapping = {
            1: "cause",
            2: "motivation",
            3: "prerequisite_location",
            4: "prerequisite_posession", 
            5: "prerequisite_other",
            6: "subseq_event",
            7: "reaction",
            8: "subseq_event_location",
            9: "subseq_event_posession",
            10: "subseq_event_other",
        }
        return domain_mapping[dimension]

    def _get_pred_and_ref_cicero(self, data, preds):
        generated_sequences, gold_sequences = {domain:[] for domain in self.domains}, {domain:[] for domain in self.domains}
        idx = 0
        for row in data:
            domain = self._get_domain_cicero(row["Question"])
            gold = row["Choices"][row["Human Written Answer"][0]]
            pred = preds[idx]

            generated_sequences[domain].append(pred)
            gold_sequences[domain].append(gold)

            if domain == "subseq_event":
                domain = "subseq_event_clipped"
                idx += 1
                pred = preds[idx]

                generated_sequences[domain].append(pred)
                gold_sequences[domain].append(gold)
            idx += 1
        return generated_sequences, gold_sequences
    
    def _get_answer_glucose(self, answer, dimension, is_test):
        if is_test:
            # three possible answers are given in test set
            answer = answer.split("****")
            # clean up multiple spaces in answer(s)
            answer = [re.sub(' +', ' ', a) for a in answer]
        else:
            answer = re.sub(' +', ' ', answer)
        separator = {
            1: " >Causes/Enables> ",
            2: " >Motivates> ",
            3: " >Enables> ",
            4: " >Enables> ",
            5: " >Enables> ",
            6: " >Causes/Enables> ",
            7: " >Causes> ",
            8: " >Results in> ",
            9: " >Results in> ",
            10: " >Results in> ",
        }
        if dimension <= 5:
            return [a.split(separator[dimension])[0] for a in answer] if is_test else answer.split(separator[dimension])[0]
        else:
            return [a.split(separator[dimension])[1] for a in answer] if is_test else answer.split(separator[dimension])[1]
    
    def _get_pred_and_ref_glucose(self, data, preds):
        is_test = "test" in self.data_file
        generated_sequences, gold_sequences = {domain:[] for domain in self.domains}, {domain:[] for domain in self.domains}
        idx = 0
        for row in data:
            if not is_test:
                if int(row["worker_quality_rating"]) < 2:
                    continue
            for i in range(1, 11):
                key = str(i) + "_specificNL"
                if row[key] == "escaped":
                    continue
                domain = self._get_domain_glucose(i)
                gold = self._get_answer_glucose(row[key], i, is_test)
                pred = preds[idx]

                generated_sequences[domain].append(pred)
                gold_sequences[domain].append(gold)

                idx += 1
        return generated_sequences, gold_sequences

    def evaluate(self):
        with open(args.data_file, "r") as f:
            data = json.load(f)

        with open(args.pred_file, "r") as f:
            preds = f.read().splitlines()

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        if self.dataset_name == "cicero":
            generated_sequences, gold_sequences = self._get_pred_and_ref_cicero(data, preds)
        else:
            generated_sequences, gold_sequences = self._get_pred_and_ref_glucose(data, preds)

        domain_scores = {}
        
        for domain in self.domains:
            assert len(gold_sequences[domain]) > 0
            domain_scores[domain] = self.compute_metrics(scorers, generated_sequences[domain], gold_sequences[domain])
        domain_scores = self.get_average_scores(domain_scores)

        with open(f'{args.out_dir}/domain_scores.json', "w") as f:
            json.dump(domain_scores, f)

    def get_average_scores(self, scores, method="equal"):
        score = defaultdict(int)
        for domain in self.domains:
            for key in scores[domain]:
                if key != "num_instance":
                    if method == "weighted":
                        score[key] += scores[domain][key] * scores[domain]["num_instance"]
                    else:
                        score[key] += scores[domain][key]/len(self.domains)  # in CICERO paper, they just take average of domain-wise/num(domains)
                else:
                    score[key] += scores[domain][key]

        if method == "weighted":
            for key, value in score.items():
                if key != "num_instance":
                     score[key] /= score["num_instance"]
                     
        scores["average"] = score
        return scores
    
    def compute_metrics(self, scorers, predictions, golds):
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
        task_scores["num_instance"] = len(golds)
        return task_scores


# python src/utils/seq2seq_evaluate.py --pred_file save/t5-base-cicero/test_generation.txt --data_file data/cicero/test.json  --out_dir save/t5-base-cicero
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="path to generation file")
    parser.add_argument("--data_file", type=str, required=True, help="path to dataset file")
    parser.add_argument("--out_dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--dataset", type=str, required=False, help="dataset choice", default="cicero", choices=["cicero", "glucose"])

    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.evaluate()