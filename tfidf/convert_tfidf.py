import os
import json
import csv
import numpy as np
from collections import defaultdict
from thefuzz import fuzz


def _get_relation(q):
    questions = {
        "What is or could be the cause of target?" : ["cause"],
        "What is or could be the prerequisite of target?" : ["prerequisite"],
        "What is the possible emotional reaction of the listener in response to target?" : ["reaction"],
        "What is or could be the motivation of target?" : ["motivation"],
        "What subsequent event happens or could happen following the target?" : ["subseq_event", "subseq_event_clipped"],
        }
    return questions[q]


def _get_dialogue(relation, dialogue, target):
    if relation != "subseq_event_clipped":
        return " ".join(dialogue).replace(": :", ":").replace("A:", "<speaker1>").replace("B:", "<speaker2>")
    else:
        utts = [u[3:] for u in dialogue]
        if target in utts:
            idx = utts.index(target)
        else:
            idx = np.argmax([fuzz.token_set_ratio(u, target) for u in utts])
        return " ".join(dialogue[:idx+1]).replace(": :", ":").replace("A:", "<speaker1>").replace("B:", "<speaker2>")



with open("data/cicero/train.json", "r") as f:
    data = json.load(f)

final = []
final_rel = defaultdict(list)
for idx, sample in enumerate(data):
    question = sample["Question"]
    dialogue = sample["Dialogue"]
    target = sample["Target"]
    choice = sample['Human Written Answer'][0]
    answer = sample['Choices'][choice]
    relations = _get_relation(question)

    for relation in relations:
        dialogue = _get_dialogue(relation, dialogue, target)

        item = {
            "id": sample["ID"] + "-" + str(idx),
            "text": dialogue + "\n" + target + "\n" + answer,
            "title": relation,
        }
        final.append(item)
        final_rel[relation].append(item)


keys = ['id','text','title']
filename = os.path.join("./tfidf/cicero", 'train_tfidf.tsv')
with open(filename, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys, delimiter='\t')
    dict_writer.writeheader()
    dict_writer.writerows(final)  

for k, v in final_rel.items():
    filename = os.path.join(f"./tfidf/cicero/{k}", f'train_{k}_tfidf.tsv')
    with open(filename, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter='\t')
        dict_writer.writeheader()
        dict_writer.writerows(v)  