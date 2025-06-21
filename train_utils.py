import copy
import random
import torch
from torch.utils.data import DataLoader

def evaluate(model, dpo_trainer, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = dpo_trainer.compute_loss(model, batch)
            # Assume batch size equals length of any tensor in batch
            batch_size = next(iter(batch.values())).size(0)
            total_loss += loss.item() * batch_size
            count += batch_size
    model.train()
    return total_loss / count if count > 0 else 0.0


# UPDATE fl_training_grad function in train_utils.py

def fl_training_grad(
    server_model,
    dpo_trainer,
    clients,
    clients_per_round,
    comm_rounds,
    client_lr,
    momentum,
    epochs,
    testloader,
    aggregation_method,
    server_opt,
    grad_agg_steps,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model.to(device)

    for rnd in range(1, comm_rounds + 1):
        print(f"\n=== Comm round {rnd} ===")
        selected = random.sample(clients, clients_per_round)

        agg_grads = None

        for client_id, client_data in enumerate(selected, 1):
            print(f"  Client {client_id}/{clients_per_round}")
            client_model = copy.deepcopy(server_model).to(device)
            client_model.train()

            optimizer = torch.optim.SGD(
                client_model.parameters(), lr=client_lr, momentum=momentum
            )

            # UPDATED: Create DataLoader for pre-tokenized data
            from data_utils import _collate_tokenized_batch
            loader = DataLoader(
                client_data,
                batch_size=8,  # Use fixed batch size
                shuffle=True,
                collate_fn=_collate_tokenized_batch  # Use our custom collate function
            )

            client_grad_sums = [torch.zeros_like(p) for p in client_model.parameters()]
            local_step = 0

            for _ in range(epochs):
                for batch in loader:
                    local_step += 1
                    
                    # UPDATED: Move pre-tokenized batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    # UPDATED: Use the batch directly with DPO trainer
                    loss = dpo_trainer.compute_loss(client_model, batch)
                    loss.backward()

                    if local_step % grad_agg_steps == 0:
                        for idx, p in enumerate(client_model.parameters()):
                            if p.grad is not None:  # Add safety check
                                client_grad_sums[idx] += p.grad.detach()
                        optimizer.zero_grad()

            pushes = max(local_step // grad_agg_steps, 1)
            client_grads = [g / pushes for g in client_grad_sums]

            if agg_grads is None:
                agg_grads = client_grads
            else:
                agg_grads = [agg + cg for agg, cg in zip(agg_grads, client_grads)]

        agg_grads = [g / clients_per_round for g in agg_grads]

        # Apply aggregated gradients
        for param, g in zip(server_model.parameters(), agg_grads):
            param.grad = g.to(device)

        if server_opt is not None:
            server_opt.step()
            server_opt.zero_grad()
        else:
            for param in server_model.parameters():
                if param.grad is not None:
                    param.data -= client_lr * param.grad

        eval_loss = evaluate(server_model, dpo_trainer, testloader, device)
        print(f"  [Round {rnd}] Eval loss: {eval_loss:.4f}")
