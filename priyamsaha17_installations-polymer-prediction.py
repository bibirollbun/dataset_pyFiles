! pip download transformers rdkit


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer.save_pretrained("chemberta_tokenizer")


from transformers import AutoModel

model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model.save_pretrained("chemberta_model")

