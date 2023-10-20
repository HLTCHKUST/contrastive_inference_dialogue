import os
import json

import datasets
from thefuzz import fuzz
import numpy as np

YOUR_LOCAL_DOWNLOAD = "data" 
SEP_TOKEN = " \\n "

_HOMEPAGE = "https://github.com/declare-lab/CICERO"

_URLs = "https://github.com/declare-lab/CICERO/blob/main/data/train.json, https://github.com/declare-lab/CICERO/blob/main/data/val.json, https://github.com/declare-lab/CICERO/blob/main/data/test.json"

_DESCRIPTION = """\
CICERO, a new dataset for dialogue reasoning with contextualized commonsense inference. 
It contains 53K inferences for five commonsense dimensions: cause, subsequent event, prerequisite, motivation, and emotional reaction collected from 5.6K dialogues. 
To show the usefulness of CICERO for dialogue reasoning, we design several challenging generative and multichoice answer selection tasks for state-of-the-art NLP models to solve.
"""

_CITATION = """\
@inproceedings{ghosal-etal-2022-cicero,
    title = "{CICERO}: A Dataset for Contextualized Commonsense Inference in Dialogues",
    author = "Ghosal, Deepanway  and
      Shen, Siqi  and
      Majumder, Navonil  and
      Mihalcea, Rada  and
      Poria, Soujanya",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.344",
    pages = "5010--5028",
    abstract = "This paper addresses the problem of dialogue reasoning with contextualized commonsense inference. We curate CICERO, a dataset of dyadic conversations with five types of utterance-level reasoning-based inferences: cause, subsequent event, prerequisite, motivation, and emotional reaction. The dataset contains 53,105 of such inferences from 5,672 dialogues. We use this dataset to solve relevant generative and discriminative tasks: generation of cause and subsequent event; generation of prerequisite, motivation, and listener{'}s emotional reaction; and selection of plausible alternatives. Our results ascertain the value of such dialogue-centric commonsense knowledge datasets. It is our hope that CICERO will open new research avenues into commonsense-based dialogue reasoning.",
}
"""


q1 = [
    "What is or could be the cause of target?",
    "What is or could be the prerequisite of target?",
    "What is the possible emotional reaction of the listener in response to target?"
]

q2 = [
    "What is or could be the motivation of target?",
    "What subsequent event happens or could happen following the target?"
]


class Cicero(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="cicero_nlg",
            version=VERSION,
            description="Load CICERO dataset for NLG task",
        ),
        datasets.BuilderConfig(
            name="cicero_nlg_contrast",
            version=VERSION,
            description="Load CICERO dataset for NLG task for contrastive learning",
        )
    ]

    DEFAULT_CONFIG_NAME = "cicero_nlg"

    def _info(self):
        if self.config.name in ["cicero_nlg", "cicero_nlg_merge"]:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "relation": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "dialogue": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            )
        elif self.config.name in ["cicero_nlg_contrast", "cicero_nlg_merge_contrast", "cicero_nlg_contrast_generated"]:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "relation": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "dialogue": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "negative_samples": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                }
            )
        else:
            raise NotImplementedError

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # my_urls = _URLs
        # data_dir = dl_manager.download_and_extract(my_urls) 
        data_dir = YOUR_LOCAL_DOWNLOAD # point to local dir to avoid downloading the dataset again
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "cicero/train.json"), 
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "cicero/val.json"), 
                    "split": "valid"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "cicero/test.json"), 
                    "split": "test"
                },
            ),
        ]

    
    def _get_relation(self, q):
        questions = {
            "What is or could be the cause of target?" : ["cause"],
            "What is or could be the prerequisite of target?" : ["prerequisite"],
            "What is the possible emotional reaction of the listener in response to target?" : ["reaction"],
            "What is or could be the motivation of target?" : ["motivation"],
            "What subsequent event happens or could happen following the target?" : ["subseq_event", "subseq_event_clipped"],
            }
        return questions[q]
    
    def _get_dialogue(self, relation, dialogue, target):
        if relation != "subseq_event_clipped":
            new_dialogue = dialogue
            return " ".join(new_dialogue).replace(": :", ":").replace("A:", "<speaker1>").replace("B:", "<speaker2>")
        else:
            utts = [u[3:] for u in dialogue]
            if target in utts:
                idx = utts.index(target)
            else:
                idx = np.argmax([fuzz.token_set_ratio(u, target) for u in utts])
            new_dialogue = dialogue[:idx+1]
            return " ".join(new_dialogue).replace(": :", ":").replace("A:", "<speaker1>").replace("B:", "<speaker2>")

    def _generate_examples(self, filepath, split):
        if self.config.name == "cicero_nlg":
            with open(filepath, "r") as f:
                data = json.load(f)
            for row in data:
                id_ = row["ID-new"]
                question = row["Question"]
                relations = self._get_relation(question)
                target = row["Target"]
                answer = row["Choices"][row["Human Written Answer"][0]]

                for relation in relations:
                    id_ = id_ + '_' + relation
                    dialogue = self._get_dialogue(relation, row["Dialogue"], target)
                    example = {
                        "id": id_,
                        "relation": relation,
                        "question": question,
                        "answer": answer,
                        "dialogue": dialogue,
                        "target": target,
                    }
                    yield id_, example
                    
        elif self.config.name == "cicero_nlg_contrast":
            with open(filepath, "r") as f:
                data = json.load(f)
            for row in data:
                id_ = row["ID-new"]
                question = row["Question"]
                relations = self._get_relation(question)
                target = row["Target"]
                answer = row["Choices"][row["Human Written Answer"][0]]
                negative_samples = []
                for i, choice in enumerate(row["Choices"]):
                    if i != row["Human Written Answer"][0]:
                        negative_samples.append(choice)

                for relation in relations:
                    id_ = id_ + '_' + relation
                    dialogue = self._get_dialogue(relation, row["Dialogue"], target)
                    example = {
                        "id": id_,
                        "relation": relation,
                        "question": question,
                        "answer": answer,
                        "dialogue": dialogue,
                        "target": target,
                        "negative_samples": negative_samples,
                    }
                    yield id_, example
                
        else:
            raise NotImplementedError