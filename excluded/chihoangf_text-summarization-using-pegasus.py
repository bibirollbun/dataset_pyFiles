!pip install transformers
!pip install sentencepiece


from transformers import PegasusForConditionalGeneration, AutoTokenizer
import torch
import json
from tqdm import tqdm


# Đọc file JSON
json_path = "/kaggle/input/arxiv-test-sent/test_oracle_sent_arxiv.json"  # Thay đường dẫn file JSON của bạn
output_file = "pegasus_citing_test_summ.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# model_ckpt = "google/pegasus-cnn_dailymail"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)


import json
from transformers import PegasusForConditionalGeneration, AutoTokenizer
import torch
# Khởi tạo mô hình Pegasus
model_name = "google/pegasus-cnn_dailymail"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


# Lấy và nối các câu trong trường "sent_text"
output_sum = {}

for key, value in tqdm(data.items(), desc="Processing Summaries"):
    if "sent_text" in value:
        sentences = " ".join(value["sent_text"])

        # Tóm tắt văn bản
        batch = tokenizer(sentences, truncation=True, padding="longest", return_tensors="pt").to(device)
        summary_ids = model.generate(**batch)
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        # ✅ Lưu vào dictionary
        output_sum[key] = {
            "pegasus_summary": summary[0] if summary else ""
        }


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_sum, f, ensure_ascii=False, indent=4)

print(f"✅ Output đã được lưu vào: {output_file}")





