import argparse
import importlib

from PIL import Image
from PIL.Image import DecompressionBombError
from einops import repeat
import more_itertools
import numpy as np
import torch
import random

import uuid
import json
import os
import random
from tqdm import tqdm
from eval_model import BaseEvalModel
from collections import defaultdict
from open_flamingo.src.flamingo import Flamingo
from huggingface_hub import hf_hub_download
from open_flamingo.eval.vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

from eval_model import BaseEvalModel

from eval_datasets_visdial import VisdialDataset, GQADataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Use key-value caching for classification evals to speed it up. Currently this doesn't underperforms for MPT models.",
)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_visdial",
    action="store_true",
    default=False,
    help="Whether to evaluate on visdial.",
)
parser.add_argument(
    "--eval_gqa",
    action="store_true",
    default=True,
    help="Whether to evaluate on gqa.",
)

# Dataset arguments
## Visdial Dataset
parser.add_argument(
    "--visdial_test_image_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--visdial_test_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--visdial_val_image_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--visdial_val_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--visdial_train_image_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--visdial_train_json_path",
    type=str,
    default=None,
)

## GQA Dataset
parser.add_argument(
    "--gqa_image_folder_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--gqa_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--gqa_val_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--gqa_test_questions_json_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--comment",
    type=str,
    help="comment about this run",
    default="vqa"
)

parser.add_argument(
    "--type",
    type=str,
    help="operation, attribute",
    default="operation",
)

parser.add_argument(
    "--prompt",
    type=str,
    help="prompt options",
    default="wo",
)


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"models.{args.model}")
    print("leftovers", leftovers)
    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_visdial:
        print("Evaluating on Visdial 1.0...")
        for shot in args.shots:
            # print("shot:",shot)
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                # print("seed: ",seed)
                vqa_score = evaluate_visdial(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="visdial",
                )

                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)

            print(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
            results["visdial"].append(
                {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
            )

    if args.eval_gqa:
        print("Evaluating on GQA...")
        for shot in args.shots:
            # print("shot:",shot)
            accuracy = []
            score = {}
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                # print("seed: ",seed)
                result = evaluate_gqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="gqa",
                )

                print(f"Shots {shot} Trial {trial} GQA acc: {result}")
                score.update(result)
                accuracy.append(score["accuracy"])

            print(f"Shots {shot} Mean Acc: {np.mean(accuracy)}")
            results["gqa"].append(
                {"shots": shot, "trials": score, "mean accuracy": np.mean(accuracy)}
            )

    # RESULTS_FILE in scipt"
    if args.results_file is not None:
        file_path = args.results_file + ".json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )
    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
    )
    return loader


def get_valid_dataset(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
    )
    return loader


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


##### Scores data structures initialization
##########################################################################################

# book to float
def toScore(b):
    return float(1 if b else 0)


# Compute average of a list
def avg(l):
    if len(l) == 0:
        return 0
    return float(sum(l)) / len(l)


def gqa_accuracy(prediction, answer):
    correct = (prediction.lower().strip() in answer.lower().strip())
    score = toScore(correct)
    return score


def evaluate_gqa(
        args: argparse.Namespace,
        eval_model: BaseEvalModel,
        seed: int = 42,
        # min_generation_length: int = 0,
        max_generation_length: int = 5,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        num_shots: int = 8,
        dataset_name: str = "gqa",
):
    """
    Evaluate a model on GQA datasets.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    if dataset_name == "gqa":
        image_test_path = args.gqa_image_folder_path
        json_test_path = args.gqa_test_questions_json_path
        image_val_path = args.gqa_image_folder_path
        json_val_path = args.gqa_val_questions_json_path
        image_train_path = args.gqa_image_folder_path
        json_train_path = args.gqa_train_questions_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    test_dataset = GQADataset(
        image_folder_path=image_test_path,
        question_path=json_test_path,
        # is_train=False,
        # dataset_range="dev_all",
        dataset_name=dataset_name
    )
    val_dataset = GQADataset(
        image_folder_path=image_val_path,
        question_path=json_val_path,
        # is_train=False,
        # dataset_range="dev_all",
        dataset_name=dataset_name
    )
    train_dataset = GQADataset(
        image_folder_path=image_train_path,
        question_path=json_train_path,
        # is_train=True,
        # dataset_range="dev_all",
        dataset_name=dataset_name
    )
    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    # EVALUATION ON WHOLE TEST SET
    test_dataloader = get_valid_dataset(test_dataset, args.batch_size)

    # EVALUATION ON SAMPLES
    # test_dataloader = prepare_eval_samples(
    #     test_dataset,
    #     args.num_samples if args.num_samples > 0 else len(test_dataset),
    #     args.batch_size,
    #     seed,
    # )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)

    predictions = []
    exist_score = []
    for batch in tqdm(
            test_dataloader,
            desc=f"Running inference {dataset_name}",
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch["image"])
        )
        batch_images = []
        batch_text = []

        if args.type == "exist":
            prompt_options = {
                "restatement": "Please consider the following question and answer only with 'yes' or 'no': ",
                "environment": "Taking the environment into account, respond to the following question only with 'yes' or 'no': ",
                # "boolean": "For the following question, answer 'yes' if the object exists, otherwise say 'no': ",
                "scenario": "Please respond with either 'yes' or 'no' to indicate whether the object exists in the following scenario:",
                # "Imagine a scenario, answer the question with 'yes' or 'no': ",
                "confident": "Please provide a 'yes' or 'no' response based on your confidence in the presence of the object mentioned in the question. Choose 'yes' if you are highly confident it's present, and 'no' if you have any doubt or lack confidence.",
                # "Respond 'yes' only if you are completely confident that the object mentioned in the question is present; otherwise, choose 'no'."
                "game": "I'd like to play a fun game with you! In this game, I will show you an image and ask a question about whether certain objects exist in the image. You need to answer 'yes' or 'no' as quickly as possible. If your answer is correct, you'll earn a certain number of points. However, if your answer is incorrect, you may lose some points. Your goal is to earn as many points as possible and see how well you perform in this visual question-answering game!",
                "roleplay": "Imagine you're an 'object existence recognizer' in a professional role. I'll display an image, and your task is to determine if specific objects are present in the image. Please respond with either 'yes' or 'no' to indicate your judgment.",
                "task": "Your task is to answer questions regarding the existence of objects in the image. We will show you an image and then ask questions about whether certain objects are present in it. You should provide a 'yes' or 'no' answer based on your understanding of the visible objects in the image."
            }

        elif args.type == "color":
            prompt_options = {
                "mystery": "Solve the color mystery: ",
                "detective": "Be a color detective and find out: ",
                "detective2": "You are now a color detective! Examine the image closely and tell me: ",
                "story": "Tell a color story: describe the color of the object by answering the following question: ",
                "exploration": "Embark on a color exploration mission: ",

                "restatement": "Please consider the following question: ",
                "environment": "Taking the environment into account, respond to: ",
                "guess": "For the following question, answer with color if it can be recognized, otherwise, make a guess: ",
                "scenario": "Imagine a scenario, answer the question: ",
                "confident": "Respond with color only if you are completely confident that the object's color can be recognized: "
            }
        else:
            raise ValueError(f"Unsupported type: {args.type}")
        print("prompt: ", args.prompt, prompt_options.get(args.prompt, ""))

        options = ["yes", "no"]

        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])
            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=prompt_options.get(args.prompt, "") + x["question"], answer=random.choice(options)
                    )
                    for x in batch_demo_samples[i]
                ]
            )

            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        process_function = (
            postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        for new_prediction, image_id, question, question_id, answer in \
                zip(new_predictions, batch["image_id"], batch["question"], batch["question_id"], batch["answer"]):
            print('pred,ans: ', new_prediction, answer)
            predictions.append({"image_id": image_id,
                                "question": question,
                                "question_id": question_id,
                                "prediction": new_prediction,
                                "answer": answer,
                                "type": args.type})

            exist_score.append(gqa_accuracy(new_prediction, answer))

    avg_accuracy = 100.0 * avg(exist_score)
    print('DEBUG AVG:', avg_accuracy)

    scores = {"accuracy": avg_accuracy}

    prompt_example = batch_text[0]

    results_path = f"results/gqa_results/{args.type}/{num_shots}/simplified_exist/{args.comment}_{seed}.json"
    print('Results_path', results_path)

    with open(results_path, "w") as f:
        # print("prediction:", predictions)
        f.write(
            json.dumps({
                "prompt": prompt_example,
                "scores": scores,
                "result": predictions},
                indent=4,
            )
        )

    return scores


def evaluate_visdial(
        args: argparse.Namespace,
        eval_model: BaseEvalModel,
        seed: int = 42,
        # min_generation_length: int = 0,
        max_generation_length: int = 5,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        num_shots: int = 8,
        dataset_name: str = "visdial",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    if dataset_name == "visdial":
        image_test_path = args.visdial_test_image_path
        json_test_path = args.visdial_test_json_path
        image_val_path = args.visdial_val_image_path
        json_val_path = args.visdial_val_json_path
        image_train_path = args.visdial_train_image_path
        json_train_path = args.visdial_train_json_path
    # elif dataset_name == "":
    #     image_test_path=args.test_image_path
    #     json_test_path=args.test_json_path
    #     image_val_path=args.val_image_path
    #     json_val_path=args.val_json_path
    #     image_train_path=args.train_image_path
    #     json_train_path=args.train_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    test_dataset = VisdialDataset(
        image_folder_path=image_test_path,
        image_json_path=json_test_path,
        # is_train=True,
        dataset_name=dataset_name
    )
    val_dataset = VisdialDataset(
        image_folder_path=image_val_path,
        image_json_path=json_val_path,
        # is_train=True,
        dataset_name=dataset_name
    )
    train_dataset = VisdialDataset(
        image_folder_path=image_train_path,
        image_json_path=json_train_path,
        # is_train=True,
        dataset_name=dataset_name
    )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    ##TODO:
    test_dataloader = get_valid_dataset(test_dataset, args.batch_size)

    # test_dataloader = prepare_eval_samples(
    #     test_dataset,
    #     args.num_samples if args.num_samples > 0 else len(test_dataset),
    #     args.batch_size,
    #     seed,
    # )
    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = []
    dialogs = []

    for batch in tqdm(
            test_dataloader,  # test_dataloader
            desc=f"Running inference {dataset_name}",
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch["image"])
        )

        batch_images = []
        batch_text = []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])
            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x["question"], answer=x["answer"]
                    )
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )
            # print("batch text:",batch_text)
            # print("################################")
        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        # print("output:",outputs[0])
        # print("################################")

        process_function = (
            postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        # for answer_options in batch["answer_options"]:
        # predictions.append({"answer_options"}:answer_options)

        for new_prediction, image, question in zip(new_predictions, batch["image_id"], batch["question"]):
            # print("DEBUG batch:", batch)
            predictions.append({"image_id": image,
                                "question": question,
                                "prediction": new_prediction})

        actuall_list = []
        right_list = []

        accuracy = 100.0 * np.sum(right_list) / len(test_dataset)

        prompt_example = batch_text[0]
        scores = {"accuracy": accuracy}

        # TODO:
        # results_path = f"results/sample/{args.comment}_{num_shots}_{seed}.json"
        results_path = f"results/visdial_results/{args.comment}_{num_shots}_{seed}.json"
        print('Results_path', results_path)
        # print("################################")

        with open(results_path, "w") as f:
            # print("prediction:", predictions)
            f.write(
                json.dumps({
                    "prompt": prompt_example,
                    "scores": scores,
                    "result": predictions},
                    indent=4,
                )
            )

    return scores


if __name__ == "__main__":
    main()
