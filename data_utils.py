import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, Subset
import torch
from typing import List, Dict, Any


def _get_preference_dataset(name: str, split: str = "train"):
    if name.lower() in ["hh-rlhf", "anthropic-hh"]:
        return load_dataset("Anthropic/hh-rlhf", split=split)
    elif name.lower() in ["ultrafeedback", "ultrafeedback-binarized"]:
        return load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    elif name.lower() in ["oasst1", "openassistant"]:
        # Convert OASST1 to preference format
        dataset = load_dataset("OpenAssistant/oasst1", split=split)
        return _convert_oasst_to_preferences(dataset)
    elif name.lower() in ["orca-dpo", "orca"]:
        return load_dataset("Intel/orca_dpo_pairs", split=split)
    else:
        raise ValueError(f"Unsupported dataset: {name}. Supported: hh-rlhf, ultrafeedback, oasst1, orca-dpo")


def _convert_oasst_to_preferences(dataset):
    preferences = []
    
    # Group by conversation thread
    conversations = {}
    for item in dataset:
        parent_id = item.get('parent_id')
        if parent_id:
            if parent_id not in conversations:
                conversations[parent_id] = []
            conversations[parent_id].append(item)
    
    # Create preference pairs from ranked responses
    for parent_id, responses in conversations.items():
        if len(responses) >= 2:
            # Sort by rank (lower rank = better)
            responses.sort(key=lambda x: x.get('rank', float('inf')))
            
            # Create pairs: best vs worst, best vs second-best, etc.
            chosen = responses[0]['text']
            for rejected_resp in responses[1:]:
                preferences.append({
                    'prompt': responses[0].get('parent_text', ''),
                    'chosen': chosen,
                    'rejected': rejected_resp['text']
                })
    
    return Dataset.from_list(preferences)


def _create_topic_based_partition(dataset, num_clients: int, alpha: float = 1.0):
    # Simple heuristic: use prompt length and first few words to create pseudo-topics
    topics = []
    for item in dataset:
        prompt = item['prompt'].lower()
        # Create pseudo-topics based on prompt characteristics
        if any(word in prompt for word in ['code', 'programming', 'python', 'function']):
            topic = 0  # coding
        elif any(word in prompt for word in ['math', 'calculate', 'equation', 'solve']):
            topic = 1  # math
        elif any(word in prompt for word in ['write', 'story', 'creative', 'poem']):
            topic = 2  # creative
        elif any(word in prompt for word in ['explain', 'what', 'how', 'why']):
            topic = 3  # explanatory
        else:
            topic = 4  # general
        topics.append(topic)
    
    topics = np.array(topics)
    num_topics = 5
    
    # Use Dirichlet distribution to assign topics to clients
    return _dirichlet_partition(topics, num_clients, alpha, num_topics)


def _dirichlet_partition(labels, num_clients, alpha, num_classes):
    idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    client_idxs = [[] for _ in range(num_clients)]

    # Sample dirichlet proportions for each class
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    for cls, idxs in enumerate(idx_by_class):
        if len(idxs) == 0:  # Skip empty classes
            continue
            
        np.random.shuffle(idxs)
        cls_prop = proportions[cls] * len(idxs)
        cls_counts = cls_prop.astype(int)
        
        # Ensure all samples are distributed
        remainder = len(idxs) - cls_counts.sum()
        if remainder > 0:
            cls_counts[-1] += remainder
        
        start = 0
        for c in range(num_clients):
            count = cls_counts[c]
            if count > 0:
                client_idxs[c].extend(idxs[start:start + count].tolist())
                start += count
    
    return client_idxs


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer=None, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Ensure consistent field names
        if 'chosen' not in item and 'response_j' in item:
            item['chosen'] = item['response_j']
        if 'rejected' not in item and 'response_k' in item:
            item['rejected'] = item['response_k']
            
        return {
            'prompt': item.get('prompt', ''),
            'chosen': item.get('chosen', ''),
            'rejected': item.get('rejected', '')
        }


def build_dataset(dataset_name: str, batch_size: int, num_clients: int, iid_alpha: float, tokenizer=None, max_length=512):
    if tokenizer is None:
        raise ValueError("Tokenizer is required for pre-tokenization (Option A)")
    
    # Load training and test datasets
    try:
        train_dataset = _get_preference_dataset(dataset_name, split="train")
        # Many preference datasets don't have explicit test splits
        try:
            test_dataset = _get_preference_dataset(dataset_name, split="test")
        except:
            # Use a portion of training data as test set
            dataset_size = len(train_dataset)
            test_size = min(1000, dataset_size // 10)  # 10% or 1000 samples, whichever is smaller
            
            indices = list(range(dataset_size))
            np.random.shuffle(indices)
            
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            
            test_dataset = train_dataset.select(test_indices)
            train_dataset = train_dataset.select(train_indices)
            
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}: {e}")

    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")

    # Partition training data across clients
    if iid_alpha <= 0:
        # IID split: randomly distribute samples
        all_idxs = np.arange(len(train_dataset))
        np.random.shuffle(all_idxs)
        client_idxs = np.array_split(all_idxs, num_clients)
        client_idxs = [idx.tolist() for idx in client_idxs]
        print(f"Using IID split with {num_clients} clients")
    else:
        # Non-IID split based on topics/domains
        client_idxs = _create_topic_based_partition(train_dataset, num_clients, iid_alpha)
        print(f"Using non-IID split with alpha={iid_alpha} and {num_clients} clients")

    # Create pre-tokenized client datasets
    clients = []
    for i, idxs in enumerate(client_idxs):
        if len(idxs) > 0:
            client_data = train_dataset.select(idxs)
            print(f"Tokenizing client {i+1} data ({len(idxs)} samples)...")
            
            # Pre-tokenize this client's data
            tokenized_client_data = make_pref_dataset_for_dpo(client_data, tokenizer, max_length)
            clients.append(tokenized_client_data)
            print(f"Client {i+1}: {len(idxs)} samples (tokenized)")
        else:
            print(f"Warning: Client {i+1} has no samples")

    # Create pre-tokenized test dataset
    print("Tokenizing test data...")
    tokenized_test_data = make_pref_dataset_for_dpo(test_dataset, tokenizer, max_length)
    
    testloader = DataLoader(
        tokenized_test_data, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=_collate_tokenized_batch
    )

    return clients, testloader


def make_pref_dataset_for_dpo(dataset, tokenizer, max_length=512):
    def tokenize_preference(examples):
        batch_size = len(examples['prompt']) if isinstance(examples['prompt'], list) else 1
        
        # Handle single example vs batch
        if not isinstance(examples['prompt'], list):
            examples = {k: [v] for k, v in examples.items()}
            batch_size = 1
        
        tokenized = {
            'chosen_input_ids': [],
            'chosen_attention_mask': [],
            'rejected_input_ids': [],
            'rejected_attention_mask': []
        }
        
        for i in range(batch_size):
            # Tokenize chosen response (prompt + chosen)
            chosen_text = examples['prompt'][i] + " " + examples['chosen'][i]
            chosen_tokens = tokenizer(
                chosen_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Tokenize rejected response (prompt + rejected)
            rejected_text = examples['prompt'][i] + " " + examples['rejected'][i]
            rejected_tokens = tokenizer(
                rejected_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            tokenized['chosen_input_ids'].append(chosen_tokens['input_ids'].squeeze())
            tokenized['chosen_attention_mask'].append(chosen_tokens['attention_mask'].squeeze())
            tokenized['rejected_input_ids'].append(rejected_tokens['input_ids'].squeeze())
            tokenized['rejected_attention_mask'].append(rejected_tokens['attention_mask'].squeeze())
        
        return tokenized
    
    # Map tokenization over the entire dataset
    tokenized_dataset = dataset.map(
        tokenize_preference,
        batched=True,
        batch_size=100,  # Process in small batches to avoid memory issues
        remove_columns=dataset.column_names,  # Remove original text columns
        desc="Tokenizing preference pairs"
    )
    
    return tokenized_dataset


def _collate_tokenized_batch(batch):
    if len(batch) == 0:
        return {}
    
    # Stack tensors for each field
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([torch.tensor(item[key]) for item in batch])
    
    return collated