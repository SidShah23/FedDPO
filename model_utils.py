# model_utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def build_model(rank: int, alpha: int, base_name: str = "mosaicml/mpt-1b-redpajama-200b"):
    # 1) Load the pretrained RedPajama model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        device_map="auto",
    )

    # 2) Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3) Return the LoRA-wrapped model
    return get_peft_model(base_model, lora_config)

def build_tokenizer(base_name: str = "mosaicml/mpt-1b-redpajama-200b"):
    tokeizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
