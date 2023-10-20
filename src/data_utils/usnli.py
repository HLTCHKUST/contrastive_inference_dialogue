import os
import csv

import datasets


YOUR_LOCAL_DOWNLOAD = "data" 

_HOMEPAGE = "https://nlp.jhu.edu/unli/"

_URLs = ""

_DESCRIPTION = """\
UNLI (Uncertain natural language inference) is a refinement of Natural Language Inference (NLI) that shifts away from categorical labels, targeting instead the direct prediction of subjective probability assessments to model subtle distinctions on the likelihood of a hypothesis conditioned on a premise.
"""

_CITATION = """\
@inproceedings{chen-etal-2020-uncertain,
    title = "Uncertain Natural Language Inference",
    author = "Chen, Tongfei  and
      Jiang, Zhengping  and
      Poliak, Adam  and
      Sakaguchi, Keisuke  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.774",
    doi = "10.18653/v1/2020.acl-main.774",
    pages = "8772--8779",
    abstract = "We introduce Uncertain Natural Language Inference (UNLI), a refinement of Natural Language Inference (NLI) that shifts away from categorical labels, targeting instead the direct prediction of subjective probability assessments. We demonstrate the feasibility of collecting annotations for UNLI by relabeling a portion of the SNLI dataset under a probabilistic scale, where items even with the same categorical label differ in how likely people judge them to be true given a premise. We describe a direct scalar regression modeling approach, and find that existing categorically-labeled NLI data can be used in pre-training. Our best models correlate well with humans, demonstrating models are capable of more subtle inferences than the categorical bin assignment employed in current NLI tasks.",
}
"""


class aNLG(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="usnli",
            version=VERSION,
            description="Load u-snli dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "usnli"

    def _info(self):
        if self.config.name == "usnli":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label_nli": datasets.Value("int32"),
                    "label": datasets.Value("float32")
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
                    "filepath": os.path.join(data_dir, "u-snli/train.csv"), 
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "u-snli/dev.csv"), 
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "u-snli/test.csv"), 
                },
            ),
        ]


    def _generate_examples(self, filepath):
        with open(filepath, "r") as f:
            reader = csv.DictReader(f, delimiter=",")
            data = []
            for row in reader:
                data.append(row)

        for idx, row in enumerate(data):
            id_ = "usnli_"+str(idx)
            example = {
                "id": id_,
                "premise": row["pre"],
                "hypothesis": row["hyp"],
                "label_nli": row["nli"],
                "label": row["unli"]
            }

            yield id_, example
