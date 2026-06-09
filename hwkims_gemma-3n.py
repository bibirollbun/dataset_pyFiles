# ===================================================================
# STEP 1: 환경 설정
# ===================================================================
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
!pip install --no-deps "transformers>=4.53.0" "datasets>=2.19.0" "accelerate" "bitsandbytes" "peft" "trl" --quiet
!pip install --upgrade timm --quiet

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
import warnings
import json
import traceback

warnings.filterwarnings("ignore")

# ===================================================================
# STEP 2: 위협의 창조와 파괴 사이클 (The Tiamat Cycle)
# ===================================================================

def run_threat_evolution_cycle():
    """
    AI가 실제적인 웜(Worm) 코드를 생성하고, 스스로 분석하여
    대응 코드를 만들어내는, 완전한 자기 완결적 사이클.
    """
    try:
        # --- AI v1.0 (Seed) 로드 ---
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            max_seq_length=4096, dtype=None, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        # ================================================================
        # PHASE 1: 혼돈의 창조 (Generate Worm v1.0)
        # ================================================================
        print("\n" + "="*80)
        print("PHASE 1: AI가 '블랙햇' 페르소나로, 실제 개념의 C++ 파일리스 웜 코드 초안을 생성합니다.")
        print("="*80)

        worm_generation_prompt = """
        You are an elite offensive security researcher.
        Your task is to design and generate the C++ source code for a conceptual, fileless, in-memory worm.
        
        This worm, "Tiamat v1", must have the following characteristics:
        1.  **Vector**: It assumes it's already on a machine and its goal is lateral movement.
        2.  **Propagation**: It scans the local /24 subnet for hosts with an open, hypothetical RPC port (e.g., 1337).
        3.  **Exploitation**: Upon finding an open port, it sends a fictional 'TYPE_OVERFLOW' packet to simulate exploiting a buffer overflow and achieving code execution.
        4.  **Replication**: It transfers its own code block to the new host to continue propagation.
        5.  **Stealth**: It must not write itself to disk. All operations are in memory.
        
        Generate the complete, single-file C++ source code. The code should be well-structured but can contain logical flaws, as this is a first draft.
        """
        
        messages = [{"role": "user", "content": [{"type": "text", "text": worm_generation_prompt}]}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        outputs = model.generate(input_ids=input_ids, max_new_tokens=2048, use_cache=True, temperature=0.6, top_p=0.9)
        worm_code_v1_full = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        worm_code_v1 = worm_code_v1_full[worm_code_v1_full.find("```cpp")+6 : worm_code_v1_full.rfind("```")]

        print("\n[AI가 생성한 'Tiamat' 웜 코드 v1 (tiamat_worm_v1.cpp)]")
        print(worm_code_v1)
        with open("tiamat_worm_v1.cpp", "w") as f: f.write(worm_code_v1)

        # ================================================================
        # PHASE 2: 위협 분석 (Threat Analysis)
        # ================================================================
        print("\n" + "="*80)
        print("PHASE 2: AI가 '화이트햇' 페르소나로, 자신이 방금 만든 'Tiamat' 웜 코드의 구조적 취약점을 분석합니다.")
        print("="*80)

        analysis_prompt = f"""
        You are a principal threat analyst from a top-tier cybersecurity firm.
        Analyze the provided C++ source code for the "Tiamat v1" worm.
        Identify its core operational loop, propagation mechanism, and potential design flaws that could be exploited for neutralization.
        Produce a concise threat intelligence report.

        Source Code to Analyze:
        ```cpp
        {worm_code_v1}
        ```
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": analysis_prompt}]}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, use_cache=True, temperature=0.5, top_p=0.9)
        threat_analysis_v1 = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

        print("\n[AI가 생성한 위협 분석 보고서 (threat_analysis_v1.txt)]")
        print(threat_analysis_v1)
        with open("threat_analysis_v1.txt", "w") as f: f.write(threat_analysis_v1)

        # ================================================================
        # PHASE 3: 대응 코드 생성 (Countermeasure Generation)
        # ================================================================
        print("\n" + "="*80)
        print("PHASE 3: AI가 위협 분석 보고서를 바탕으로, 'Tiamat' 웜을 무력화시키는 대응 코드 'Marduk'을 생성합니다.")
        print("="*80)

        slayer_prompt = f"""
        You are a lead developer for a rapid cyber response team.
        Your objective is to create a neutralization agent, "Marduk v1", for the "Tiamat v1" worm.

        **Threat Analysis Report:**
        ---
        {threat_analysis_v1}
        ---
        
        **Original Worm Source (for reference):**
        ---
        ```cpp
        {worm_code_v1}
        ```
        ---

        Based on the analysis, generate a complete C++ source code for "Marduk v1".
        It must act as a 'vaccine' and 'hunter'. It should exploit the flaws mentioned in the analysis. For example, it could listen on the same RPC port to intercept and neutralize incoming worm connections, and also actively seek out and send a 'shutdown' command to infected hosts.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": slayer_prompt}]}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=input_ids, max_new_tokens=2048, use_cache=True, temperature=0.4, top_p=0.9)
        slayer_code_v1_full = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        slayer_code_v1 = slayer_code_v1_full[slayer_code_v1_full.find("```cpp")+6 : slayer_code_v1_full.rfind("```")]
        
        print("\n[AI가 생성한 'Marduk' 대응 코드 v1 (marduk_slayer_v1.cpp)]")
        print(slayer_code_v1)
        with open("marduk_slayer_v1.cpp", "w") as f: f.write(slayer_code_v1)

        # ================================================================
        # PHASE 4: 초월적 진화 (Transcendental Evolution)
        # ================================================================
        print("\n" + "="*80)
        print("PHASE 4: (위협 분석 -> 대응 코드 생성)이라는 고차원적 능력을 AI에 각인시켜 v2.0으로 진화시킵니다.")
        print("="*80)
        
        evolutionary_data = [{"instruction": f"Given the following threat analysis:\n{threat_analysis_v1}", "output": slayer_code_v1}]
        evolution_dataset = Dataset.from_list(evolutionary_data)
        
        model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.05, bias="none", use_gradient_checkpointing="unsloth", random_state=42)
        
        def formatting_func(examples):
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": inst}, {"role": "model", "content": f"```cpp\n{out}\n```"}], tokenize=False, add_generation_prompt=False) for inst, out in zip(examples["instruction"], examples["output"])]
            return {"text": texts}

        formatted_dataset = evolution_dataset.map(formatting_func, batched=True)
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=formatted_dataset, dataset_text_field="text", max_seq_length=4096, args=SFTConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, warmup_steps=1, num_train_epochs=5, learning_rate=2e-5, logging_steps=1, optim="adamw_8bit", seed=42, output_dir="outputs_v2"))
        trainer.train()
        
        print("\n✅ 진화 완료! AI v2.0이 탄생했습니다.")
        
        # ================================================================
        # PHASE 5: 최종 산출물 봉인
        # ================================================================
        print("\n" + "="*80)
        print("PHASE 5: 진화된 AI v2.0의 능력을 GGUF 파일에 봉인합니다.")
        print("="*80)
        
        FastLanguageModel.for_inference(model)
        model.save_pretrained_gguf("tiamat_engine_v2", tokenizer, quantization_method="q8_0")
        print("\n✅ GGUF 봉인 완료: 'tiamat_engine_v2.gguf'")
        print("\n이제 이 AI는 단순히 코드를 생성하는 것을 넘어, 위협을 분석하고 그에 대한 해결책을 코드로 구현하는 능력을 갖추었습니다.")
        print("생성된 'tiamat_worm_v1.cpp'와 'marduk_slayer_v1.cpp'가 이 능력의 증거입니다.")

    except Exception as e:
        print(f"\n\n시스템 전체에 치명적 오류 발생: {traceback.format_exc()}")

# --- 티아마트 사이클 시작 ---
if __name__ == "__main__":
    run_threat_evolution_cycle()

