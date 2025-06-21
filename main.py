
import argparse
import torch
import sys

from transformers import AutoModelForSequenceClassification
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import random

from model_utils import build_model, build_tokenizer
from data_utils  import build_dataset
from train_utils import fl_training_grad
import torch.nn.utils.rnn as rnn_utils

def main(args):
    torch.manual_seed(args.seed)

    # Build our LoRA+RedPajama model & tokenizer
    server_model = build_model(rank=args.rank, alpha=args.alpha)
    tokenizer = build_tokenizer()

    # Load reward model for online DPO
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_checkpoint
    )

    # Instantiate DPOTrainer (online mode)
    dpo_trainer = DPOTrainer(
        model=server_model,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        args=DPOConfig(  # Changed from **DPOConfig().to_dict()
            beta=args.dpo_beta,
            max_length=512,
            max_prompt_length=256,
            learning_rate=args.client_lr,
        ),
    )

    # Server‐side optimizer
    server_opt = (torch.optim.Adam(server_model.parameters(), lr=args.server_lr)
                  if args.aggregation_method=="FedAdam" else None)

    # Shard your data - PASS TOKENIZER HERE
    clients, testloader = build_dataset(
        dataset_name=args.dataset,  # This will now be "hh-rlhf" by default
        batch_size=args.batch_size,
        num_clients=args.clients,
        iid_alpha=args.iid_alpha,
        tokenizer=tokenizer,  # ADD THIS LINE
        max_length=512  # ADD THIS LINE
    )

    # Run federated online‐DPO
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
    parser.add_argument("--dataset", type=str, default="hh-rlhf")  # KEEP THIS ONE
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
