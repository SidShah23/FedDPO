from datasets import Dataset


def make_pref_dataset(raw_prefs, tokenizer, max_length=512):
    ds = Dataset.from_list(raw_prefs)

    def tokenize_pref(example):
        chosen = tokenizer(
            example['prompt'], example['chosen'],
            truncation=True, max_length=max_length,
        )
        rejected = tokenizer(
            example['prompt'], example['rejected'],
            truncation=True, max_length=max_length,
        )
        return {
            'chosen_input_ids': chosen['input_ids'],
            'chosen_attention_mask': chosen['attention_mask'],
            'rejected_input_ids': rejected['input_ids'],
            'rejected_attention_mask': rejected['attention_mask'],
        }

    pref_ds = ds.map(
        tokenize_pref,
        remove_columns=['prompt', 'chosen', 'rejected'],
        batched=False,
    )
    return pref_ds
