from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def build_model(rank: int, alpha: int, base_name: str = "distilgpt2"):
    # Use DistilGPT2 - only ~80MB vs 8GB for MPT
    base_model = AutoModelForCausalLM.from_pretrained(base_name)

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    try:
        return get_peft_model(base_model, lora_config)
    except ValueError as e:
        print(f"LoRA failed: {e}, trying all-linear")
        lora_config.target_modules = "all-linear"
        try:
            return get_peft_model(base_model, lora_config)
        except:
            print("Using base model without LoRA")
            return base_model

def build_tokenizer(base_name: str = "distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer