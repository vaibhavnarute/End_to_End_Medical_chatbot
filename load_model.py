from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
