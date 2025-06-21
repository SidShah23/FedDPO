import copy
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model, dpo_trainer, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    
    # Add progress bar for evaluation
    eval_pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in eval_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = dpo_trainer.compute_loss(model, batch)
            # Assume batch size equals length of any tensor in batch
            batch_size = next(iter(batch.values())).size(0)
            total_loss += loss.item() * batch_size
            count += batch_size
            
            # Update progress bar with current loss
            eval_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    model.train()
    return total_loss / count if count > 0 else 0.0


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

    # Main progress bar for communication rounds
    comm_pbar = tqdm(range(1, comm_rounds + 1), desc="Communication Rounds", position=0)
    
    for rnd in comm_pbar:
        comm_pbar.set_description(f"Comm Round {rnd}/{comm_rounds}")
        
        selected = random.sample(clients, clients_per_round)
        agg_grads = None

        # Progress bar for clients in current round
        client_pbar = tqdm(enumerate(selected, 1), 
                          total=len(selected), 
                          desc="Training Clients", 
                          position=1, 
                          leave=False)

        for client_id, client_data in client_pbar:
            client_pbar.set_description(f"Client {client_id}/{clients_per_round}")
            
            client_model = copy.deepcopy(server_model).to(device)
            client_model.train()

            optimizer = torch.optim.SGD(
                client_model.parameters(), lr=client_lr, momentum=momentum
            )

            # Create DataLoader for pre-tokenized data
            from data_utils import _collate_tokenized_batch
            loader = DataLoader(
                client_data,
                batch_size=8,
                shuffle=True,
                collate_fn=_collate_tokenized_batch
            )

            client_grad_sums = [torch.zeros_like(p) for p in client_model.parameters()]
            local_step = 0

            # Progress bar for epochs
            epoch_pbar = tqdm(range(epochs), 
                            desc=f"Client {client_id} Epochs", 
                            position=2, 
                            leave=False)

            for epoch in epoch_pbar:
                epoch_pbar.set_description(f"Client {client_id} Epoch {epoch+1}/{epochs}")
                
                # Progress bar for batches within each epoch
                batch_pbar = tqdm(loader, 
                                desc=f"Batches", 
                                position=3, 
                                leave=False)
                
                epoch_loss = 0.0
                batch_count = 0
                
                for batch in batch_pbar:
                    local_step += 1
                    batch_count += 1
                    
                    # Move pre-tokenized batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}

                    # Use the batch directly with DPO trainer
                    loss = dpo_trainer.compute_loss(client_model, batch)
                    loss.backward()
                    
                    epoch_loss += loss.item()

                    if local_step % grad_agg_steps == 0:
                        for idx, p in enumerate(client_model.parameters()):
                            if p.grad is not None:
                                client_grad_sums[idx] += p.grad.detach()
                        optimizer.zero_grad()
                    
                    # Update batch progress bar
                    avg_loss = epoch_loss / batch_count
                    batch_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}', 
                        'avg_loss': f'{avg_loss:.4f}',
                        'step': local_step
                    })
            
            # Clean up progress bars
            epoch_pbar.close()

            pushes = max(local_step // grad_agg_steps, 1)
            client_grads = [g / pushes for g in client_grad_sums]

            if agg_grads is None:
                agg_grads = client_grads
            else:
                agg_grads = [agg + cg for agg, cg in zip(agg_grads, client_grads)]

            # Update client progress with final info
            client_pbar.set_postfix({
                'local_steps': local_step,
                'pushes': pushes
            })

        client_pbar.close()

        # Aggregate gradients
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

        # Evaluate and update main progress bar
        print(f"\n[Round {rnd}] Starting evaluation...")
        eval_loss = evaluate(server_model, dpo_trainer, testloader, device)
        
        comm_pbar.set_postfix({
            'eval_loss': f'{eval_loss:.4f}',
            'clients': clients_per_round
        })
        
        print(f"[Round {rnd}] Eval loss: {eval_loss:.4f}")

    comm_pbar.close()
    print("Federated training completed!")