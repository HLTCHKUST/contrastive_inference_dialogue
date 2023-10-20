# modified from https://github.com/nlpcl-lab/dialog-eval-hard-negative/blob/ddf0aa65be4d26d65659587fdc25aeab527e0594/mask_and_fill_by_bert.py
import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, RobertaForMaskedLM, set_seed
from datasets import load_dataset


SEP_TOKEN = " \\n "
softmax = torch.nn.Softmax(dim=0)


def prepare_input_texts(row, dataset_config_name="cicero_nlg"):
    answer = row["answer"]
    if dataset_config_name == "cicero_nlg":
        dialogue = row["dialogue"] + " " + answer
    elif dataset_config_name == "cicero_mcq_single_select":
        # answer = row["choice"+str(row["label"])]
        dialogue = (SEP_TOKEN.join([row["question"], "target: " + row["target"], "context: " + row["dialogue"]]), answer)
    elif dataset_config_name == "cicero_cls_single_select":
        # other_choices = [row["choice"+str(i)] for i in range(5) if i != row["label"]]
        other_choices = row["negative_samples"]
        context = SEP_TOKEN.join([row["question"], "target: " + row["target"], "context: " + row["dialogue"]])
        context = context + SEP_TOKEN + "other choices: "
        context += SEP_TOKEN.join(other_choices)
        dialogue = (context, answer)
    return dialogue, answer


def main(args):
    # check configuration
    if args.model_name_or_path == "/home/xuyan/CommonsenseDialogue/CoINF/save/roberta_cls_single":
        assert args.dataset_config_name == "cicero_cls_single_select"
    elif args.model_name_or_path == "/home/xuyan/CommonsenseDialogue/CoINF/save/roberta_selection_sep":
        assert args.dataset_config_name == "cicero_mcq_single_select"
    # load dataset
    dataset = load_dataset(args.dataset_name, "cicero_nlg_contrast")[args.split]

    # load moodel and tokenizer
    model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # check whether we have to compute scores or not
    skip_get_score = False
    if args.score_file_path:
        skip_get_score = True
        with open(args.score_file_path, "r") as f:
            saved_scores = json.load(f)
        assert len(dataset) == len(saved_scores)
    
    if args.num_return_samples > 1:
        assert args.top_k > 1

    # main process!
    generated_negatives = []
    for i, row in enumerate(tqdm(dataset)):
        # get scores
        dialogue, answer = prepare_input_texts(row, dataset_config_name=args.dataset_config_name)
        gens = {"id": row["id"], "generated_negatives": [], "answer": answer}
        encoded_dialog = tokenizer([dialogue], return_tensors="pt", max_length=args.max_length, truncation=True)
        encoded_response = tokenizer(answer, return_tensors="pt", max_length=args.max_length, truncation=True)
        response_begin_index_in_dialog = (len(encoded_dialog["input_ids"][0]) - len(encoded_response["input_ids"][0]) + 1)
        encoded_dialog = {k: v.to(args.device) for k, v in encoded_dialog.items()}
        encoded_response = {k: v.to(args.device) for k, v in encoded_response.items()}
        word_list = []
        if not skip_get_score:
            dialog_score_list, response_score_list, diff_score_list = [], [], []
            for response_token_index in range(len(encoded_response["input_ids"][0]) - 2):
                response_index_in_dialog = response_token_index + response_begin_index_in_dialog
                response_index_in_response = response_token_index + 1

                # Find the original token and check the integrity
                original_token_in_dialog = (
                    encoded_dialog["input_ids"][0][response_index_in_dialog].clone().detach()
                )
                original_token_in_response = (
                    encoded_response["input_ids"][0][response_index_in_response]
                    .clone()
                    .detach()
                )
                word_list.append(tokenizer.convert_ids_to_tokens([original_token_in_dialog])[0])

                # Mask the current token in both dialog and response sentence
                encoded_dialog["input_ids"][0][
                    response_index_in_dialog
                ] = tokenizer.mask_token_id
                encoded_response["input_ids"][0][
                    response_index_in_response
                ] = tokenizer.mask_token_id

                with torch.no_grad():
                    dialog_output = model(**encoded_dialog)[0]
                    response_output = model(**encoded_response)[0]

                score_in_dialog = float(
                    softmax(dialog_output[0][response_index_in_dialog])[
                        original_token_in_dialog
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )
                score_in_response = float(
                    softmax(response_output[0][response_index_in_response])[
                        original_token_in_response
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )

                diff_score_list.append(np.log(score_in_dialog) - np.log(score_in_response))
                dialog_score_list.append(np.log(score_in_dialog))
                response_score_list.append(np.log(score_in_response))
        else:
            diff_score_list = saved_scores[i]["score"]
            for response_token_index in range(len(encoded_response["input_ids"][0]) - 2):
                response_index_in_dialog = response_token_index + response_begin_index_in_dialog
                response_index_in_response = response_token_index + 1

                # Find the original token and check the integrity
                original_token_in_dialog = (
                    encoded_dialog["input_ids"][0][response_index_in_dialog].clone().detach()
                )
                original_token_in_response = (
                    encoded_response["input_ids"][0][response_index_in_response]
                    .clone()
                    .detach()
                )
                word_list.append(tokenizer.convert_ids_to_tokens([original_token_in_dialog])[0])

        gens["score"] = diff_score_list

        # get generation
        encoded = tokenizer(answer, return_tensors="pt", max_length=args.max_length, truncation=True,)
        assert len(encoded["input_ids"][0]) == len(word_list) + 2

        masked_token_indices = []
        masked_token_original_list = []
        for idx, score in enumerate(diff_score_list):
            if score >= args.threashold:
                masked_token_original_list.append(
                    encoded["input_ids"][0][idx + 1].clone().detach()
                )
                encoded["input_ids"][0][idx + 1] = tokenizer.mask_token_id
                masked_token_indices.append(idx + 1)
        gens["threashold"] = args.threashold

        if args.dynamic and len(masked_token_indices) == 0:
            # force to replace some of the token so that we have a negative sample for all
            threashold = args.threashold + 0.25
            while len(masked_token_indices) == 0:
                for idx, score in enumerate(diff_score_list):
                    if score >= threashold:
                        masked_token_original_list.append(
                            encoded["input_ids"][0][idx + 1].clone().detach()
                        )
                        encoded["input_ids"][0][idx + 1] = tokenizer.mask_token_id
                        masked_token_indices.append(idx + 1)
                threashold -= 0.25
            gens["threashold"] = threashold

        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)[0]

        for n in range(args.num_return_samples):
            changed_indices = []
            for mask_order, mask_index in enumerate(masked_token_indices):
                vocab = softmax(output[0][mask_index])
                while True:
                    if args.num_return_samples == 1:
                        decoded_index = torch.argmax(vocab).item()
                    else:
                        top_k_indices = torch.topk(vocab, args.top_k).indices.tolist()
                        decoded_index = random.choice(top_k_indices)

                    if decoded_index not in [masked_token_original_list[mask_order]]:
                        break
                    output[0][mask_index, decoded_index] = -100
                changed_indices.append(decoded_index)

            for idx, mask_position in enumerate(masked_token_indices):
                encoded["input_ids"][0][mask_position] = changed_indices[idx]

            changed_response = tokenizer.decode(
                encoded["input_ids"][0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            gens["generated_negatives"].append(changed_response)
        generated_negatives.append(gens)
        if args.debug:
            print("original:", answer)
            print("generated:", "\n".join(gens["generated_negatives"]))
            # print("orig_gen:", saved_scores[i]["generated_negatives"])
            # print("threashold:", gens["threashold"])
            input()
    
    # save generation
    save_file_path = f"{args.save_path}/{args.split}_generated_negatives_{args.threashold}_roberta-large_{args.num_return_samples}.json"
    if args.dynamic:
        save_file_path = save_file_path.replace(".json", "_dynamic.json")
    with open(save_file_path, "w") as f:
        json.dump(generated_negatives, f)


#  python src/utils/generate_negatives.py -cu 4 --save_path save/debug --split validation --score_file_path /nfs/etsuko/CoINF/save/roberta_cls_single/validation_generated_negatives.json --debug
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-large",
        required=False,
        help="Pre-trained model name or path")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="path to save generated negatives")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        required=False,
        help="Maximum length")
    parser.add_argument(
        "--score_file_path",
        type=str,
        required=False,
        default=None,
        help="path to a saved score file for generation")
    parser.add_argument('--dataset_name', type=str, default="src/data_utils/cicero.py")
    parser.add_argument('--dataset_config_name', type=str, default="cicero_nlg")
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation", required=False)
    parser.add_argument("--threashold", type=float, default=0.5, required=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dynamic", action="store_true", help="Allow dynamic threshold to generate negatives for all the samples")
    parser.add_argument("--num_return_samples", type=int, default=1, help="How many negative samples to return per sample")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--top_k", type=int, default=1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)
    main(args)
