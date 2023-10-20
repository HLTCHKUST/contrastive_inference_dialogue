# Contrastive Learning for Inference in Dialogue

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="img/caire.png" width="20%"> <img align="right" src="img/HKUST.jpeg" width="12%">

The implementation of the paper "Contrastive Learning for Inference in Dialogue":

**Contrastive Learning for Inference in Dialogue**. [Etsuko Ishii](https://etsukokuste.github.io/), [Yan Xu](https://yana-xuyan.github.io), Bryan Wilie, Ziwei Ji, Holy Lovenia, Willy Chung, and Pascale Fung. **EMNLP2023** [[PDF]](https://arxiv.org/pdf/2310.12467.pdf)

If you use any source codes included in this toolkit in your work, please cite the following paper:

<pre>
@article{ishii2023contrastive,
  title={Contrastive Learning for Inference in Dialogue},
  author={Ishii, Etsuko and Xu, Yan and Wilie, Bryan and Ji, Ziwei and Lovenia, Holy and Chung, Willy and Fung, Pascale},
  journal={arXiv preprint arXiv:2310.12467},
  year={2023}
}
</pre>


## Environment
Python 3.8 and Pytorch 1.11.0, and the other packages follow `requirements.txt`.
Please download dependencies with `pip install -r requirements.txt`.


## Datasets
We use the [CICERO](https://github.com/declare-lab/CICERO) datasets in our main experiments. You can find the data under `data/cicero/{train/val/test}.json`.
We also offer the difficulty annotation for the CICERO test set based on the semantic information (`data/cicero/annoted_results.json`). The difficulty score annotation corresponds to `1: Sufficient, 2: Likely, 3: Conceivable` in the paper.


## How to run the code
### Contrastive Learning
1. To train the model with and without contrastive learning, run
```bash
sh run_trainer.sh
```

2. To evaluate the model trained, run e.g.,
```bash
python evaluate_trainer.py --model_name_or_path save/t5-base-cicero-contrast --save_path save/t5-base-cicero-contrast --cu 0 --n_beam 5 --dataset_name src/data_utils/cicero.py --dataset_config cicero_nlg --bs 16
```

### 3-shot Prompting with LLMs
If you want to use prompts manually sampled in `src/utils/cicero_prompt.py`, run e.g.,:
```bash
CUDA_VISIBLE_DEVICES=0 python cicero_prompt.py --k 3 --model_name_or_path EleutherAI/gpt-j-6B --seed 42 --save_path save/gptj-ciero-3shot-seed42
```

If you want to use tf-idf sampler, run e.g.,:
```bash
CUDA_VISIBLE_DEVICES=0 python cicero_prompt_tfidf.py --topk 3 --model_name_or_path EleutherAI/gpt-j-6B --save_path save/gptj-ciero-3shot-tfidf
```
Note that you first have to obtain a tf-idf retriever. For more information, please refer to https://github.com/efficientqa/retrieval-based-baselines.


### Evaluating with NLI models
For the UNLI model, train the Roberta-large by:
```bash
sh run_nli.sh
```
Please download the UNLI data from [here](https://nlp.jhu.edu/unli/) and locate under `data/u-snli`.
To obtain the scores reported in the paper, run:
```bash
python src/utils/get_usnli_score.py -cu 0 --hypothesis_file_path save/t5-base-cicero/test_generation.txt --bs 4 --save_path save/results/t5-base-cicero
```

For the AlignScore, please refer to the [original repo by the authors](https://github.com/yuh-zha/AlignScore).
