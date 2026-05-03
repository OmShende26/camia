# Objective: compute the probability over each token as the features for each point in the dataset and save the results in a pkl file
# Modified for custom JSONL dataset from colab

import torch
from tqdm import tqdm
import math
from collections import defaultdict
import time
from utils import *
import os
from mimir.config import ExperimentConfig
from omegaconf import DictConfig
import hydra
import os
import pickle
import json
import numpy as np
from typing import List
from datasets import load_dataset 

from mimir.utils import fix_seed
from mimir.models_without_debugging import LanguageModel



def load_jsonl_dataset(jsonl_path):
    """
    Load texts from JSONL file.
    Expected format: {"text": "..."}
    """
    texts = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    texts.append(obj.get('text', ''))
    except Exception as e:
        print(f"Error loading {jsonl_path}: {e}")
    return texts


def load_wikimia_dataset(repo_id="SimMIA/WikiMIA-25", split="WikiMIA_length64"):
    """
    Load texts from Hugging Face WikiMIA benchmark.
    Expected columns: "input" (text) and "label" (1 for member, 0 for non-member)
    """
    print(f"Downloading/Loading HF Dataset: {repo_id} (Split: {split})...")
    try:
        dataset = load_dataset(repo_id, split=split)
    except Exception as e:
        print(f"Error loading Hugging Face dataset {repo_id}: {e}")
        return [], []
        
    member_texts =[]
    nonmember_texts =[]
    
    for row in dataset:
        # In WikiMIA: label == 1 indicates seen/member, label == 0 indicates unseen/nonmember
        if row.get("label") == 1:
            member_texts.append(row.get("input", ""))
        else:
            nonmember_texts.append(row.get("input", ""))
            
    return member_texts, nonmember_texts




def get_probability_history(
    data,
    target_model: LanguageModel,
    config: ExperimentConfig,
    n_samples: int = 100,
    batch_size: int = 100,
):
    """Extract token probability features from texts"""
    num_repeatitions = 5
    fix_seed(config.random_seed)
    n_samples = len(data["records"]) if n_samples is None else n_samples
    results = []
    
    for batch in tqdm(
        range(math.ceil(n_samples / batch_size)), desc=f"Computing criterion"
    ):
        texts = data["records"][batch * batch_size : (batch + 1) * batch_size]

        # For each entry in batch
        for idx in range(len(texts)):
            sample_information = defaultdict(list)
            sample = (
                texts[idx][: config.max_substrs] if config.full_doc else [texts[idx]]
            )

            for idx, substr in enumerate(sample):
                # Skip very short texts
                if len(substr.split()) < 3:
                    continue
                    
                try:
                    s_tk_probs, s_all_probs, labels = (
                        target_model.get_probabilities_with_tokens(
                            substr, return_all_probs=True
                        )
                    )
                    sample_information["tk_probs"].append(s_tk_probs)
                    sample_information["labels"].append(labels)

                    # Consider repeating the input text
                    all_str = substr
                    for r_idx in range(num_repeatitions):
                        all_str = all_str + " " + substr
                    s_tk_probs, s_all_probs, labels = (
                        target_model.get_probabilities_with_tokens(
                            all_str, return_all_probs=True
                        )
                    )
                    sample_information[f"tk_probs_repeated_{num_repeatitions}"].append(
                        s_tk_probs
                    )
                    sample_information[f"labels_repeated_{num_repeatitions}"].append(labels)
                except Exception as e:
                    print(f"Error processing text: {e}")
                    continue
                    
            if sample_information:  # Only add if we got results
                results.append(sample_information)

    return results


def generate_data_processed(
    config, raw_data_member, batch_size: int, raw_data_non_member: List[str] = None
):
    """Prepare member and non-member text pairs"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    data = {
        "nonmember": [],
        "member": [],
    }

    seq_lens = []
    
    # Ensure both have same length
    min_len = min(len(raw_data_member), len(raw_data_non_member))
    raw_data_member = raw_data_member[:min_len]
    raw_data_non_member = raw_data_non_member[:min_len]
    
    num_batches = (min_len // batch_size) + 1
    iterator = tqdm(range(num_batches), desc="Preparing data")
    
    for batch in iterator:
        member_text = raw_data_member[batch * batch_size : (batch + 1) * batch_size]
        non_member_text = raw_data_non_member[
            batch * batch_size : (batch + 1) * batch_size
        ]

        for o, s in zip(non_member_text, member_text):
            if not config.full_doc:
                seq_lens.append((len(s.split(" ")), len(o.split())))

            if config.tok_by_tok:
                for tok_cnt in range(len(o.split(" "))):
                    data["nonmember"].append(" ".join(o.split(" ")[: tok_cnt + 1]))
                    data["member"].append(" ".join(s.split(" ")[: tok_cnt + 1]))
            else:
                data["nonmember"].append(o)
                data["member"].append(s)

    n_samples = len(data["nonmember"])
    return data, seq_lens, n_samples


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fix_seed(config.random_seed)

    exp_name = "camia_custom_dataset"
    sf = os.path.join(exp_name, config.base_model.replace("/", "_"))
    new_folder = os.path.join(config.env_config.results, sf)
    
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    print(f"Saving results to: {os.path.abspath(new_folder)}")
    
    # Setup cache
    cache_dir = config.env_config.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir: {cache_dir}")

    # Load model
    print(f"Loading model: {config.base_model}")
    base_model = LanguageModel(config)
    if not config.env_config.device_map:
        base_model.to(device)

    # # # Load custom JSONL datasets from colab paths
    # print("Loading datasets from colab paths...")
    # member_path = "/kaggle/input/datasets/omvijayshende/camia-dataset/seen_books.jsonl"
    # nonmember_path = "/kaggle/input/datasets/omvijayshende/camia-dataset/unseen_books.jsonl"    

    USE_HF_DATASET = False  # Set to True to load from Hugging Face, False to load from local JSONL files
    
    if USE_HF_DATASET:
        # WikiMIA-25 splits are: "WikiMIA_length32", "WikiMIA_length64", "WikiMIA_length128"
        data_member, data_nonmember = load_wikimia_dataset(
            repo_id="SimMIA/WikiMIA-25", 
            split="WikiMIA_length64" 
        )
        print(f"Loaded {len(data_member)} member texts from Hugging Face.")
        print(f"Loaded {len(data_nonmember)} non-member texts from Hugging Face.")
    
    else:
        # Load local JSONL datasets 
        member_path = "/kaggle/input/datasets/omvijayshende/final-seen-unseen-books-ds/Seen_train_dataset.jsonl"
        nonmember_path = "/kaggle/input/datasets/omvijayshende/final-seen-unseen-books-ds/unseen_30_train_dataset.jsonl"    
        print("Hi, I am using books dataset")
        print(f"Loading member data from {member_path}...")
        data_member = load_jsonl_dataset(member_path)
        print(f"Loaded {len(data_member)} member texts")
        
        print(f"Loading non-member data from {nonmember_path}...")
        data_nonmember = load_jsonl_dataset(nonmember_path)
        print(f"Loaded {len(data_nonmember)} non-member texts")
    # ==========================================


    if not data_member or not data_nonmember:
        raise ValueError("Failed to load datasets. Check file paths.")

    # Prepare data
    data, seq_lens, n_samples = generate_data_processed(
        config,
        data_member,
        batch_size=config.batch_size,
        raw_data_non_member=data_nonmember,
    )

    print(f"Prepared {n_samples} sample pairs")

    data_members = {"records": data["member"]}
    data_nonmembers = {"records": data["nonmember"]}

    # Extract features for members
    print("Extracting features for members...")
    member_features = get_probability_history(
        data_members,
        target_model=base_model,
        config=config,
        n_samples=n_samples,
        batch_size=config.batch_size,
    )
    
    # Extract features for non-members
    print("Extracting features for non-members...")
    nonmember_features = get_probability_history(
        data_nonmembers,
        target_model=base_model,
        config=config,
        n_samples=n_samples,
        batch_size=config.batch_size,
    )

    # Save features
    output_file = os.path.join(new_folder, "all_features_seen_unseen_books.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(
            {"member_preds": member_features, "nonmember_preds": nonmember_features}, f
        )

    print(f"Saved features to {output_file}")


if __name__ == "__main__":
    main()
