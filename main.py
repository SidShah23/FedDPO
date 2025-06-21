import argparse
import torch
import sys
import os

from transformers import AutoModelForSequenceClassification
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import random

from model_utils import build_model, build_tokenizer
from data_utils import build_dataset
from train_utils import fl_training_grad

def main(args):
    torch.manual_seed(args.seed)
    
    # Set environment variables to avoid tokenizer warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Build model & tokenizer
    server_model = build_model(rank=args.rank, alpha=args.alpha)
    tokenizer = build_tokenizer()

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Load dataset first
    print("Loading and preparing dataset...")
    clients, testloader = build_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_clients=args.clients,
        iid_alpha=args.iid_alpha,
        tokenizer=tokenizer,
        max_length=512
    )

    # Create proper dummy dataset for DPO trainer (with prompt field)
    dummy_data = [{
        "prompt": "Human: Hello\n\nAssistant:",
        "chosen": " Hi there! How can I help you today?",
        "rejected": " Go away, I'm busy."
    }]
    dummy_dataset = Dataset.from_list(dummy_data)

    # Create output directory
    os.makedirs("./dpo_output", exist_ok=True)

    # DPO trainer with proper dummy dataset
    dpo_trainer = DPOTrainer(
        model=server_model,
        train_dataset=dummy_dataset,  # Now has proper structure
        eval_dataset=None,
        tokenizer=tokenizer,
        args=DPOConfig(
            output_dir="./dpo_output",
            beta=args.dpo_beta,
            max_length=512,
            max_prompt_length=256,
            learning_rate=args.client_lr,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            logging_steps=10,
            save_steps=1000,
            eval_steps=1000,
            remove_unused_columns=False,
            dataloader_num_workers=0,
        ),
    )

    # Server optimizer
    server_opt = (torch.optim.Adam(server_model.parameters(), lr=args.server_lr)
                  if args.aggregation_method=="FedAdam" else None)

    print("Starting federated training...")
    fl_training_grad(
        server_model=server_model,
        dpo_trainer=dpo_trainer,
        clients=clients,
        clients_per_round=args.clients_round,
        comm_rounds=args.comm_rounds,
        client_lr=args.client_lr,
        momentum=args.momentum,
        epochs=args.epochs,
        testloader=testloader,
        aggregation_method=args.aggregation_method,
        server_opt=server_opt,
        grad_agg_steps=args.grad_agg_steps,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Online-DPO with LoRA",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # System
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    # Data & FL
    parser.add_argument("--dataset", type=str, default="hh-rlhf")
    parser.add_argument("--clients",      type=int, default=5)
    parser.add_argument("--iid-alpha",    type=float, default=-1.0)
    parser.add_argument("--clients-round",type=int, default=2)
    parser.add_argument("--comm-rounds",  type=int, default=2)
    parser.add_argument("--batch-size",   type=int, default=8)

    # LoRA
    parser.add_argument("--rank",  type=int,   default=4)
    parser.add_argument("--alpha", type=int,   default=4)

    # DPO
    parser.add_argument("--dpo-beta",                type=float, default=0.1)
    parser.add_argument(
        "--reward-model-checkpoint", type=str,
        default="facebook/bart-large-mnli",
        help="SequenceClassification model for online DPO"
    )

    # Server / aggregation
    parser.add_argument(
        "--aggregation-method", type=str,
        choices=["FedAdam","SGD"], default="FedAdam"
    )
    parser.add_argument("--server-lr", type=float, default=1e-2)

    # Client
    parser.add_argument("--client-lr",      type=float, default=1e-2)
    parser.add_argument("--momentum",       type=float, default=0.9)
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument(
        "--grad-agg-steps", type=int, default=1,
        help="Mini-batches per gradient aggregation"
    )

    args = parser.parse_args()
    main(args)