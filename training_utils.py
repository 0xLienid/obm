import torch
from torch.utils.data import DataLoader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tokenize_batch(examples, tokenizer, max_seq_len):
    input_ids = tokenizer.encode_batch(examples["text"])
    input_ids = [ids[:max_seq_len] for ids in input_ids]

    targets = []
    attn_masks = []
    for i, id in enumerate(input_ids):
        seq_len = len(id)

        if seq_len < 10:
            return None

        if seq_len < max_seq_len:
            id = id + [50256] * (max_seq_len - seq_len)

        input_ids[i] = id
        targets.append(input_ids[i][1:] + [50256])
        assert len(input_ids[i]) == len(targets[i])

        attn_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        attn_masks.append(attn_mask)

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device="cpu")
    targets_tensor = torch.tensor(targets, dtype=torch.long, device="cpu")
    attn_masks_tensor = torch.tensor(
        attn_masks, dtype=torch.long, device="cpu")

    return input_ids_tensor, targets_tensor, attn_masks_tensor


def tokenize_dataset(dataset, tokenizer, max_seq_len, batch_size, num_batches):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batches = []
    batch_i = 0
    for batch in dataloader:
        result = tokenize_batch(
            batch, tokenizer, max_seq_len)

        if result is None:
            continue

        input_ids, target_ids, attn_masks = result
        batch_dict = {
            "inputs": input_ids,
            "targets": target_ids,
            "attn_masks": attn_masks
        }
        batches.append(batch_dict)

        batch_i += 1
        if batch_i >= num_batches:
            break

    return batches
