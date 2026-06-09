# ==============================================================================
# ULTIMATE ENSEMBLE: Structured Cascade + Advanced ML Models
# ==============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# è®¾ç½®éš�æœºç§�å­�
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ==============================================================================
# 1. æ•°æ�®é›†ç±» - ç»“æ�„åŒ–æ��ç¤ºè¯�å¤„ç�†
# ==============================================================================
class StructuredPromptDataset(Dataset):
    """
    ç»“æ�„åŒ–æ��ç¤ºè¯�æ•°æ�®é›†
    æ”¯æŒ�çœŸå®�åŒºåŸŸæ ‡ç­¾ç”Ÿæˆ�å’Œæ–‡æœ¬ç»“æ�„æ˜ å°„
    """

    def __init__(self, df, tokenizer, max_length=512, mode="train"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt, section_info = self._build_structured_prompt_with_mapping(row)

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "row_id": row["row_id"],
        }

        if self.mode == "train" and "rule_violation" in row:
            item["violation_label"] = torch.tensor(
                row["rule_violation"], dtype=torch.float
            )
            # ç”Ÿæˆ�çœŸå®�çš„åŒºåŸŸæ ‡ç­¾
            item["section_labels"] = self._generate_real_section_labels(
                prompt, encoding, section_info
            )

        return item

    def _build_structured_prompt_with_mapping(self, row):
        """æ�„å»ºç»“æ�„åŒ–æ��ç¤ºè¯�å¹¶è®°å½•çœŸå®�çš„æ–‡æœ¬æ®µè�½æ˜ å°„"""

        # æ�„å»ºå�„ä¸ªéƒ¨åˆ†çš„æ–‡æœ¬
        task_section = "Task: Determine if a comment violates the given rule."
        rule_section = f"Rule: {row['rule']}"
        pos_examples_section = f"Examples of VIOLATING comments:\n1. {row['positive_example_1']}\n2. {row['positive_example_2']}"
        neg_examples_section = f"Examples of NON-VIOLATING comments:\n1. {row['negative_example_1']}\n2. {row['negative_example_2']}"
        comment_section = f"Comment to evaluate: {row['body']}"
        context_section = f"Context: Posted in r/{row.get('subreddit', 'unknown')}\n\nDoes this comment violate the rule?"

        # ç»„è£…å®Œæ•´æ��ç¤ºè¯�å¹¶è®°å½•æ¯�éƒ¨åˆ†çš„å­—ç¬¦ä½�ç½®
        sections = [
            task_section,
            rule_section,
            pos_examples_section,
            neg_examples_section,
            comment_section,
            context_section,
        ]
        section_names = [
            "task",
            "rule",
            "positive_examples",
            "negative_examples",
            "comment",
            "context",
        ]

        # è®¡ç®—æ¯�ä¸ªéƒ¨åˆ†åœ¨å®Œæ•´æ–‡æœ¬ä¸­çš„å­—ç¬¦ä½�ç½®
        full_prompt = ""
        section_char_ranges = {}

        for i, (section_name, section_text) in enumerate(zip(section_names, sections)):
            start_pos = len(full_prompt)
            if i > 0:  # é™¤äº†ç¬¬ä¸€ä¸ªsectionï¼Œéƒ½è¦�åŠ æ�¢è¡Œ
                full_prompt += "\n\n"
                start_pos = len(full_prompt)

            full_prompt += section_text
            end_pos = len(full_prompt)
            section_char_ranges[section_name] = (start_pos, end_pos)

        return full_prompt, section_char_ranges

    def _generate_real_section_labels(self, prompt, encoding, section_char_ranges):
        """åŸºäº�çœŸå®�çš„æ–‡æœ¬ç»“æ�„ç”Ÿæˆ�åŒºåŸŸæ ‡ç­¾"""

        # è�·å�–tokenizerçš„è¾“å‡º
        input_ids = encoding["input_ids"].flatten()
        seq_len = len(input_ids)

        # åˆ›å»ºæ ‡ç­¾çŸ©é˜µ [seq_len, 6]
        section_labels = torch.zeros(seq_len, 6, dtype=torch.float)

        # è�·å�–æ¯�ä¸ªtokenå¯¹åº”çš„å­—ç¬¦èŒƒå›´
        token_char_spans = self._get_token_char_spans(prompt, encoding)

        # ä¸ºæ¯�ä¸ªåŒºåŸŸåˆ†é…�æ ‡ç­¾
        section_names = [
            "task",
            "rule",
            "positive_examples",
            "negative_examples",
            "comment",
            "context",
        ]

        for section_idx, section_name in enumerate(section_names):
            if section_name in section_char_ranges:
                char_start, char_end = section_char_ranges[section_name]

                # æ‰¾åˆ°å±�äº�è¿™ä¸ªåŒºåŸŸçš„æ‰€æœ‰tokens
                for token_idx, (token_start, token_end) in enumerate(token_char_spans):
                    if token_idx >= seq_len:
                        break

                    # è®¡ç®—tokenä¸�sectionçš„é‡�å� ç¨‹åº¦
                    overlap_start = max(char_start, token_start)
                    overlap_end = min(char_end, token_end)

                    if overlap_start < overlap_end:  # æœ‰é‡�å� 
                        overlap_ratio = (overlap_end - overlap_start) / max(
                            token_end - token_start, 1
                        )
                        section_labels[token_idx, section_idx] = overlap_ratio

        # å½’ä¸€åŒ–ï¼šæ¯�ä¸ªtokençš„æ‰€æœ‰sectionæ�ƒé‡�å’Œä¸º1
        row_sums = section_labels.sum(dim=1, keepdim=True)
        section_labels = section_labels / (row_sums + 1e-9)

        return section_labels

    def _get_token_char_spans(self, text, encoding):
        """è�·å�–æ¯�ä¸ªtokenåœ¨å�Ÿå§‹æ–‡æœ¬ä¸­çš„å­—ç¬¦èŒƒå›´"""

        input_ids = encoding["input_ids"].flatten()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        char_spans = []
        current_pos = 0

        for i, token in enumerate(tokens):
            # è·³è¿‡ç‰¹æ®Štoken
            if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]:
                char_spans.append((current_pos, current_pos))
                continue

            # å¤„ç�†subword tokens
            if token.startswith("##") or token.startswith("â–�"):
                clean_token = token[2:] if token.startswith("##") else token[1:]
            else:
                clean_token = token

            # åœ¨å‰©ä½™æ–‡æœ¬ä¸­æŸ¥æ‰¾è¿™ä¸ªtoken
            remaining_text = text[current_pos:]
            token_start_in_remaining = self._find_token_in_text(
                clean_token, remaining_text
            )

            if token_start_in_remaining != -1:
                token_start = current_pos + token_start_in_remaining
                token_end = token_start + len(clean_token)
                current_pos = token_end
            else:
                # å¦‚æ�œæ‰¾ä¸�åˆ°ç²¾ç¡®åŒ¹é…�ï¼Œä½¿ç”¨å½“å‰�ä½�ç½®
                token_start = current_pos
                token_end = current_pos + max(len(clean_token), 1)
                current_pos = token_end

            char_spans.append((token_start, token_end))

        return char_spans

    def _find_token_in_text(self, token, text):
        """åœ¨æ–‡æœ¬ä¸­æŸ¥æ‰¾tokençš„ä½�ç½®ï¼Œæ”¯æŒ�å¤§å°�å†™ä¸�æ•�æ„Ÿå’Œéƒ¨åˆ†åŒ¹é…�"""

        clean_token = token.strip().lower()
        clean_text = text.lower()

        if not clean_token:
            return -1

        # ç›´æ�¥åŒ¹é…�
        pos = clean_text.find(clean_token)
        if pos != -1:
            return pos

        # å¦‚æ�œæ˜¯æ ‡ç‚¹ç¬¦å�·æˆ–å�•å­—ç¬¦ï¼Œå°�è¯•åœ¨å¼€å¤´æŸ¥æ‰¾
        if len(clean_token) == 1:
            for i, char in enumerate(clean_text):
                if char == clean_token:
                    return i

        return -1


# ==============================================================================
# 2. æ·±åº¦å­¦ä¹ æ¨¡å�‹ - å¢�å¼ºç»“æ�„åŒ–çº§è�”æ¨¡å�‹
# ==============================================================================
class EnhancedStructuredCascadeModel(nn.Module):
    """
    å¢�å¼ºçš„ç»“æ�„åŒ–çº§è�”æ¨¡å�‹
    åŒ…å�«åŒºåŸŸè¯†åˆ«ã€�æ³¨æ„�åŠ›æœºåˆ¶ã€�ç›¸ä¼¼åº¦è®¡ç®—å’Œçº§è�”å†³ç­–
    """

    def __init__(self, model_name="microsoft/deberta-v3-large", hidden_dim=1024):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self._freeze_embeddings()

        # === ç»“æ�„åŒ–åŒºåŸŸè¯†åˆ« ===
        self.section_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 6),  # 6ä¸ªåŒºåŸŸçš„logits
        )

        # === åŒºåŸŸæ„ŸçŸ¥çš„æ³¨æ„�åŠ›æœºåˆ¶ ===
        self.rule_attention = nn.MultiheadAttention(
            hidden_dim, 8, dropout=0.1, batch_first=True
        )
        self.example_attention = nn.MultiheadAttention(
            hidden_dim, 8, dropout=0.1, batch_first=True
        )
        self.comment_attention = nn.MultiheadAttention(
            hidden_dim, 8, dropout=0.1, batch_first=True
        )

        # === çº§è�”ç›¸ä¼¼åº¦è®¡ç®— ===
        self.similarity_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # === çº§è�”å†³ç­–ç½‘ç»œ ===
        # ç¬¬ä¸€çº§ï¼šåŒºåŸŸè��å�ˆ
        self.region_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),  # 6ä¸ªåŒºåŸŸ
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ç¬¬äºŒçº§ï¼šç›¸ä¼¼åº¦æ•´å�ˆ
        self.similarity_integration = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),  # åŒºåŸŸç‰¹å¾� + 6ä¸ªç›¸ä¼¼åº¦ç‰¹å¾�
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # ç¬¬ä¸‰çº§ï¼šæœ€ç»ˆå†³ç­–
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def _freeze_embeddings(self):
        """å†»ç»“åµŒå…¥å±‚å�‚æ•°ï¼Œå‡�å°‘è¿‡æ‹Ÿå�ˆ"""
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(self.encoder, "deberta") and hasattr(
            self.encoder.deberta, "embeddings"
        ):
            for param in self.encoder.deberta.embeddings.parameters():
                param.requires_grad = False
        else:
            for name, param in self.encoder.named_parameters():
                if "embedding" in name.lower():
                    param.requires_grad = False

        frozen_params = sum(
            1 for param in self.encoder.parameters() if not param.requires_grad
        )
        total_params = sum(1 for param in self.encoder.parameters())
        print(
            f"   ğŸ”’ Frozen {frozen_params}/{total_params} encoder parameters (embeddings)"
        )

    def forward(self, input_ids, attention_mask, section_labels=None):
        batch_size = input_ids.size(0)

        # === åŸºç¡€ç¼–ç � ===
        encoder_outputs = self.encoder(
            input_ids, attention_mask, output_hidden_states=True
        )
        hidden_states = (
            encoder_outputs.last_hidden_state
        )  # [batch_size, seq_len, hidden_dim]

        # === ç¬¬ä¸€é˜¶æ®µï¼šç»“æ�„åŒ–åŒºåŸŸé¢„æµ‹ ===
        section_logits = self.section_predictor(
            hidden_states
        )  # [batch_size, seq_len, 6]
        section_probs = torch.softmax(section_logits, dim=-1)

        # === ç¬¬äºŒé˜¶æ®µï¼šåŸºäº�çœŸå®�æ ‡ç­¾çš„åŒºåŸŸè¡¨ç¤ºæ��å�– ===
        if section_labels is not None and self.training:
            section_weights = section_labels  # è®­ç»ƒæ—¶ä½¿ç”¨çœŸå®�æ ‡ç­¾
        else:
            section_weights = section_probs  # æ�¨ç�†æ—¶ä½¿ç”¨é¢„æµ‹çš„æ¦‚ç�‡

        # æ��å�–å�„åŒºåŸŸè¡¨ç¤º
        regions = self._extract_regional_representations_enhanced(
            hidden_states, section_weights, attention_mask
        )

        # === ç¬¬ä¸‰é˜¶æ®µï¼šåŒºåŸŸé—´çº§è�”å…³ç³»å»ºæ¨¡ ===
        # è§„åˆ™å¢�å¼ºï¼šè§„åˆ™åŒºåŸŸå…³æ³¨å…¨å±€ä¸Šä¸‹æ–‡
        rule_enhanced, _ = self.rule_attention(
            query=regions["rule"].unsqueeze(1),
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=~attention_mask.bool(),
        )
        rule_enhanced = rule_enhanced.squeeze(1)

        # ç¤ºä¾‹å¯¹æ¯”ï¼šæ­£è´Ÿä¾‹ä¹‹é—´çš„å¯¹æ¯”å­¦ä¹ 
        example_stack = torch.stack(
            [regions["positive_examples"], regions["negative_examples"]], dim=1
        )  # [batch_size, 2, hidden_dim]

        example_enhanced, example_attn = self.example_attention(
            query=example_stack, key=example_stack, value=example_stack
        )

        # è¯„è®ºç�†è§£ï¼šè¯„è®ºå…³æ³¨è§„åˆ™å’Œç¤ºä¾‹
        comment_context = torch.cat(
            [rule_enhanced.unsqueeze(1), example_enhanced], dim=1
        )

        comment_enhanced, comment_attn = self.comment_attention(
            query=regions["comment"].unsqueeze(1),
            key=comment_context,
            value=comment_context,
        )
        comment_enhanced = comment_enhanced.squeeze(1)

        # === ç¬¬å››é˜¶æ®µï¼šçº§è�”ç›¸ä¼¼åº¦è®¡ç®— ===
        # æŠ•å½±åˆ°ç›¸ä¼¼åº¦ç©ºé—´
        rule_sim = self.similarity_projector(rule_enhanced)
        pos_sim = self.similarity_projector(example_enhanced[:, 0, :])
        neg_sim = self.similarity_projector(example_enhanced[:, 1, :])
        comment_sim = self.similarity_projector(comment_enhanced)

        # è®¡ç®—å¤šç»´ç›¸ä¼¼åº¦ç‰¹å¾�
        rule_comment_sim = F.cosine_similarity(rule_sim, comment_sim, dim=-1)
        pos_comment_sim = F.cosine_similarity(pos_sim, comment_sim, dim=-1)
        neg_comment_sim = F.cosine_similarity(neg_sim, comment_sim, dim=-1)
        rule_pos_sim = F.cosine_similarity(rule_sim, pos_sim, dim=-1)
        rule_neg_sim = F.cosine_similarity(rule_sim, neg_sim, dim=-1)
        pos_neg_sim = F.cosine_similarity(pos_sim, neg_sim, dim=-1)

        similarity_features = torch.stack(
            [
                rule_comment_sim,
                pos_comment_sim,
                neg_comment_sim,
                rule_pos_sim,
                rule_neg_sim,
                pos_neg_sim,
            ],
            dim=-1,
        )

        # === ç¬¬äº”é˜¶æ®µï¼šçº§è�”å†³ç­– ===
        # çº§è�”ç¬¬ä¸€å±‚ï¼šåŒºåŸŸè��å�ˆ
        all_regions = torch.cat(
            [
                regions["task"],
                regions["rule"],
                regions["positive_examples"],
                regions["negative_examples"],
                regions["comment"],
                regions["context"],
            ],
            dim=-1,
        )
        region_features = self.region_fusion(all_regions)

        # çº§è�”ç¬¬äºŒå±‚ï¼šç›¸ä¼¼åº¦æ•´å�ˆ
        cascade_input = torch.cat([region_features, similarity_features], dim=-1)
        cascade_features = self.similarity_integration(cascade_input)

        # çº§è�”ç¬¬ä¸‰å±‚ï¼šæœ€ç»ˆåˆ†ç±»
        violation_logits = self.final_classifier(cascade_features)

        return {
            "violation_logits": violation_logits,
            "section_logits": section_logits,
            "section_probs": section_probs,
            "similarity_features": similarity_features,
            "regional_features": regions,
            "cascade_features": cascade_features,
            "attention_weights": {
                "example_attention": example_attn,
                "comment_attention": comment_attn,
            },
        }

    def _extract_regional_representations_enhanced(
        self, hidden_states, section_weights, attention_mask
    ):
        """åŸºäº�çœŸå®�sectionæ�ƒé‡�æ��å�–åŒºåŸŸè¡¨ç¤º"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        regions = {}
        region_names = [
            "task",
            "rule",
            "positive_examples",
            "negative_examples",
            "comment",
            "context",
        ]

        for i, region_name in enumerate(region_names):
            # è�·å�–è¯¥åŒºåŸŸçš„æ�ƒé‡� [batch_size, seq_len]
            region_weights = section_weights[:, :, i]

            # åº”ç”¨attention mask
            region_weights = region_weights * attention_mask.float()

            # å½’ä¸€åŒ–æ�ƒé‡�
            region_weights = region_weights / (
                region_weights.sum(dim=-1, keepdim=True) + 1e-9
            )

            # åŠ æ�ƒå¹³å�‡æ± åŒ–
            region_repr = torch.sum(
                hidden_states * region_weights.unsqueeze(-1), dim=1
            )  # [batch_size, hidden_dim]

            regions[region_name] = region_repr

        return regions


# ==============================================================================
# 3. æ·±åº¦å­¦ä¹ è®­ç»ƒå™¨
# ==============================================================================
class EnhancedCascadeTrainer:
    """
    å¢�å¼ºçº§è�”æ¨¡å�‹è®­ç»ƒå™¨
    æ”¯æŒ�å¤šæ�Ÿå¤±å‡½æ•°ã€�æ¢¯åº¦ç´¯ç§¯å’Œæ—©å�œ
    """

    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps=4,
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # ä¼˜åŒ–å™¨å’Œè°ƒåº¦å™¨
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-8
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-6
        )

        # æ�Ÿå¤±å‡½æ•°
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

        self.accumulated_steps = 0

    def train_epoch(self, dataloader):
        """è®­ç»ƒä¸€ä¸ªepoch"""
        self.model.train()
        total_loss = 0
        violation_preds, violation_targs = [], []
        accumulated_loss = 0
        self.accumulated_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            violation_labels = batch["violation_label"].to(self.device)
            section_labels = batch["section_labels"].to(self.device)

            # å‰�å�‘ä¼ æ’­
            outputs = self.model(input_ids, attention_mask, section_labels)

            # === çº§è�”æ�Ÿå¤±è®¡ç®— ===
            # 1. ä¸»è¦�è¿�è§„åˆ†ç±»æ�Ÿå¤±
            violation_logits = outputs["violation_logits"]
            if violation_logits.dim() > 1:
                violation_logits = violation_logits.squeeze(-1)
            violation_loss = self.bce_loss(violation_logits, violation_labels)

            # 2. ç»“æ�„æ„ŸçŸ¥æ�Ÿå¤±ï¼šé¢„æµ‹çš„sectionæ¦‚ç�‡ vs çœŸå®�sectionæ ‡ç­¾
            section_logits = outputs["section_logits"]  # [batch_size, seq_len, 6]
            section_probs = torch.log_softmax(section_logits, dim=-1)

            # è®¡ç®—KLæ•£åº¦æ�Ÿå¤±
            section_loss = self.kl_div_loss(
                section_probs.view(-1, 6), section_labels.view(-1, 6)
            )

            # 3. ç›¸ä¼¼åº¦ä¸€è‡´æ€§æ�Ÿå¤±
            similarity_features = outputs["similarity_features"]
            similarity_targets = self._compute_similarity_targets(violation_labels)
            similarity_loss = self.mse_loss(similarity_features, similarity_targets)

            # === æ€»æ�Ÿå¤± ===
            total_batch_loss = (
                1.0 * violation_loss  # ä¸»ä»»åŠ¡
                + 0.5 * section_loss  # ç»“æ�„æ„ŸçŸ¥
                + 0.3 * similarity_loss  # ç›¸ä¼¼åº¦ä¸€è‡´æ€§
            )

            loss = total_batch_loss / self.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()
            violation_preds.extend(violation_logits.detach().cpu().numpy())
            violation_targs.extend(violation_labels.detach().cpu().numpy())

            # æ¢¯åº¦æ›´æ–°
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += accumulated_loss
                accumulated_loss = 0
                self.accumulated_steps += 1

                if self.accumulated_steps % 20 == 0:
                    avg_loss = total_loss / self.accumulated_steps
                    print(
                        f"    Step {self.accumulated_steps}, Avg Loss: {avg_loss:.4f}"
                    )

        self.scheduler.step()

        avg_loss = total_loss / max(self.accumulated_steps, 1)
        auc = (
            roc_auc_score(violation_targs, violation_preds)
            if len(set(violation_targs)) > 1
            else 0.5
        )

        return avg_loss, auc

    def _compute_similarity_targets(self, violation_labels):
        """åŸºäº�è¿�è§„æ ‡ç­¾è®¡ç®—ç›¸ä¼¼åº¦ç›®æ ‡"""
        batch_size = violation_labels.size(0)
        similarity_targets = torch.zeros(batch_size, 6, device=violation_labels.device)

        for i, label in enumerate(violation_labels):
            if label > 0.5:  # è¿�è§„æ ·æœ¬
                # [rule_comment, pos_comment, neg_comment, rule_pos, rule_neg, pos_neg]
                similarity_targets[i] = torch.tensor(
                    [0.8, 0.7, 0.3, 0.6, 0.2, 0.1], device=violation_labels.device
                )
            else:  # é��è¿�è§„æ ·æœ¬
                similarity_targets[i] = torch.tensor(
                    [0.4, 0.3, 0.7, 0.4, 0.6, 0.2], device=violation_labels.device
                )

        return similarity_targets

    def evaluate(self, dataloader):
        """è¯„ä¼°æ¨¡å�‹"""
        self.model.eval()
        preds, targs = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                violation_labels = batch["violation_label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                violation_logits = outputs["violation_logits"]

                if violation_logits.dim() > 1:
                    violation_logits = violation_logits.squeeze(-1)

                preds.extend(violation_logits.cpu().numpy())
                targs.extend(violation_labels.cpu().numpy())

        return roc_auc_score(targs, preds) if len(set(targs)) > 1 else 0.5

    def predict(self, dataloader):
        """é¢„æµ‹æµ‹è¯•é›†"""
        self.model.eval()
        preds, row_ids = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                violation_logits = outputs["violation_logits"]

                if violation_logits.dim() > 1:
                    violation_logits = violation_logits.squeeze(-1)

                preds.extend(violation_logits.cpu().numpy())
                row_ids.extend(batch["row_id"].numpy())

        return row_ids, preds


# ==============================================================================
# 4. æœºå™¨å­¦ä¹ ç‰¹å¾�å·¥ç¨‹
# ==============================================================================
class AdvancedMLFeatureExtractor:
    """
    é«˜çº§æœºå™¨å­¦ä¹ ç‰¹å¾�æ��å�–å™¨
    åŒ…å�«æ–‡æœ¬ã€�ç»Ÿè®¡ã€�ç›¸ä¼¼åº¦å’Œé«˜çº§è¯­è¨€ç‰¹å¾�
    """

    def __init__(self):
        # TF-IDFå�‘é‡�åŒ–å™¨
        self.tfidf_rule = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_comment = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_examples = TfidfVectorizer(
            max_features=500, stop_words="english", ngram_range=(1, 2)
        )

        # æƒ…æ„Ÿè¯�å…¸
        self.sentiment_words_positive = set(
            [
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "love",
                "like",
            ]
        )
        self.sentiment_words_negative = set(
            [
                "bad",
                "terrible",
                "awful",
                "horrible",
                "disgusting",
                "hate",
                "stupid",
                "idiot",
            ]
        )

        # å¼ºçƒˆè¯�æ±‡
        self.intensity_words = set(
            [
                "very",
                "extremely",
                "totally",
                "completely",
                "absolutely",
                "really",
                "super",
            ]
        )

    def fit_transform(self, df):
        """æ��å�–è®­ç»ƒç‰¹å¾�å¹¶æ‹Ÿå�ˆè½¬æ�¢å™¨"""
        print("    ğŸ“Š Extracting TF-IDF features...")

        # === æ–‡æœ¬ç‰¹å¾� ===
        rule_tfidf = self.tfidf_rule.fit_transform(df["rule"].fillna(""))
        comment_tfidf = self.tfidf_comment.fit_transform(df["body"].fillna(""))

        # å�ˆå¹¶æ­£è´Ÿä¾‹
        examples_text = (
            df["positive_example_1"].fillna("")
            + " "
            + df["positive_example_2"].fillna("")
            + " "
            + df["negative_example_1"].fillna("")
            + " "
            + df["negative_example_2"].fillna("")
        )
        examples_tfidf = self.tfidf_examples.fit_transform(examples_text)

        print("    ğŸ“ˆ Extracting statistical features...")

        # === ç»Ÿè®¡ç‰¹å¾� ===
        stat_features = self._extract_statistical_features(df)

        print("    ğŸ”— Extracting similarity features...")

        # === ç›¸ä¼¼åº¦ç‰¹å¾� ===
        similarity_features = self._extract_similarity_features(df)

        print("    ğŸ§  Extracting advanced features...")

        # === é«˜çº§ç‰¹å¾� ===
        advanced_features = self._extract_advanced_features(df)

        # å�ˆå¹¶æ‰€æœ‰ç‰¹å¾�
        all_features = np.hstack(
            [
                rule_tfidf.toarray(),
                comment_tfidf.toarray(),
                examples_tfidf.toarray(),
                stat_features,
                similarity_features,
                advanced_features,
            ]
        )

        return all_features

    def transform(self, df):
        """è½¬æ�¢æµ‹è¯•ç‰¹å¾�"""
        rule_tfidf = self.tfidf_rule.transform(df["rule"].fillna(""))
        comment_tfidf = self.tfidf_comment.transform(df["body"].fillna(""))

        examples_text = (
            df["positive_example_1"].fillna("")
            + " "
            + df["positive_example_2"].fillna("")
            + " "
            + df["negative_example_1"].fillna("")
            + " "
            + df["negative_example_2"].fillna("")
        )
        examples_tfidf = self.tfidf_examples.transform(examples_text)

        stat_features = self._extract_statistical_features(df)
        similarity_features = self._extract_similarity_features(df)
        advanced_features = self._extract_advanced_features(df)

        all_features = np.hstack(
            [
                rule_tfidf.toarray(),
                comment_tfidf.toarray(),
                examples_tfidf.toarray(),
                stat_features,
                similarity_features,
                advanced_features,
            ]
        )

        return all_features

    def _extract_statistical_features(self, df):
        """æ��å�–ç»Ÿè®¡ç‰¹å¾�"""
        features = []

        for _, row in df.iterrows():
            rule = str(row["rule"])
            comment = str(row["body"])
            pos1 = str(row["positive_example_1"])
            pos2 = str(row["positive_example_2"])
            neg1 = str(row["negative_example_1"])
            neg2 = str(row["negative_example_2"])

            feat = [
                # é•¿åº¦ç‰¹å¾�
                len(rule.split()),
                len(comment.split()),
                len(pos1.split()),
                len(pos2.split()),
                len(neg1.split()),
                len(neg2.split()),
                # å­—ç¬¦ç‰¹å¾�
                len(rule),
                len(comment),
                # æ¯”ä¾‹ç‰¹å¾�
                len(comment.split()) / max(len(rule.split()), 1),
                len(comment) / max(len(rule), 1),
                # å¤§å†™å­—æ¯�æ¯”ä¾‹
                sum(1 for c in comment if c.isupper()) / max(len(comment), 1),
                # æ ‡ç‚¹ç¬¦å�·æ¯”ä¾‹
                sum(1 for c in comment if not c.isalnum() and not c.isspace())
                / max(len(comment), 1),
                # é—®å�·æ„Ÿå�¹å�·æ•°é‡�
                comment.count("?"),
                comment.count("!"),
                # æ˜¯å�¦åŒ…å�«è„�è¯�ç­‰å…³é”®è¯�
                int(
                    any(
                        word in comment.lower()
                        for word in ["fuck", "shit", "damn", "stupid", "idiot"]
                    )
                ),
                # æ•°å­—å’Œç‰¹æ®Šå­—ç¬¦ç»Ÿè®¡
                sum(1 for c in comment if c.isdigit()),
                comment.count("@"),
                comment.count("#"),
                comment.count("http"),
            ]

            features.append(feat)

        return np.array(features)

    def _extract_similarity_features(self, df):
        """æ��å�–ç›¸ä¼¼åº¦ç‰¹å¾�"""
        features = []

        for _, row in df.iterrows():
            rule = str(row["rule"])
            comment = str(row["body"])
            pos1 = str(row["positive_example_1"])
            pos2 = str(row["positive_example_2"])
            neg1 = str(row["negative_example_1"])
            neg2 = str(row["negative_example_2"])

            # ç®€å�•è¯�é‡�å� ç›¸ä¼¼åº¦
            rule_words = set(rule.lower().split())
            comment_words = set(comment.lower().split())
            pos1_words = set(pos1.lower().split())
            pos2_words = set(pos2.lower().split())
            neg1_words = set(neg1.lower().split())
            neg2_words = set(neg2.lower().split())

            feat = [
                # è¯„è®ºä¸�è§„åˆ™çš„è¯�é‡�å� 
                len(rule_words & comment_words)
                / max(len(rule_words | comment_words), 1),
                # è¯„è®ºä¸�æ­£ä¾‹çš„è¯�é‡�å� 
                len(pos1_words & comment_words)
                / max(len(pos1_words | comment_words), 1),
                len(pos2_words & comment_words)
                / max(len(pos2_words | comment_words), 1),
                # è¯„è®ºä¸�è´Ÿä¾‹çš„è¯�é‡�å� 
                len(neg1_words & comment_words)
                / max(len(neg1_words | comment_words), 1),
                len(neg2_words & comment_words)
                / max(len(neg2_words | comment_words), 1),
                # æ­£ä¾‹ä¸�è´Ÿä¾‹çš„è¯�é‡�å� 
                len(pos1_words & neg1_words) / max(len(pos1_words | neg1_words), 1),
                len(pos2_words & neg2_words) / max(len(pos2_words | neg2_words), 1),
                # å¹³å�‡æ­£ä¾‹ä¸�è¯„è®ºçš„ç›¸ä¼¼åº¦
                (
                    len(pos1_words & comment_words)
                    / max(len(pos1_words | comment_words), 1)
                    + len(pos2_words & comment_words)
                    / max(len(pos2_words | comment_words), 1)
                )
                / 2,
                # å¹³å�‡è´Ÿä¾‹ä¸�è¯„è®ºçš„ç›¸ä¼¼åº¦
                (
                    len(neg1_words & comment_words)
                    / max(len(neg1_words | comment_words), 1)
                    + len(neg2_words & comment_words)
                    / max(len(neg2_words | comment_words), 1)
                )
                / 2,
            ]

            features.append(feat)

        return np.array(features)

    def _extract_advanced_features(self, df):
        """æ��å�–é«˜çº§ç‰¹å¾�"""
        features = []

        for _, row in df.iterrows():
            rule = str(row["rule"]).lower()
            comment = str(row["body"]).lower()
            pos1 = str(row["positive_example_1"]).lower()
            pos2 = str(row["positive_example_2"]).lower()
            neg1 = str(row["negative_example_1"]).lower()
            neg2 = str(row["negative_example_2"]).lower()

            comment_words = comment.split()

            feat = [
                # æƒ…æ„Ÿç‰¹å¾�
                sum(1 for word in self.sentiment_words_positive if word in comment),
                sum(1 for word in self.sentiment_words_negative if word in comment),
                # å¼ºåº¦è¯�ç‰¹å¾�
                sum(1 for word in self.intensity_words if word in comment),
                # é‡�å¤�å­—ç¬¦ç‰¹å¾�
                self._count_repeated_chars(comment),
                # å…¨å¤§å†™å�•è¯�ç‰¹å¾�
                sum(1 for word in comment.split() if word.isupper() and len(word) > 1),
                # è¯­è¨€å¤�æ�‚åº¦
                len(set(comment_words)) / max(len(comment_words), 1),  # è¯�æ±‡å¤šæ ·æ€§
                # å¹³å�‡è¯�é•¿
                np.mean([len(word) for word in comment_words]) if comment_words else 0,
                # ä¸�å­�ç‰ˆå�—çš„ç›¸å…³æ€§
                int("politics" in str(row.get("subreddit", "")).lower()),
                int("news" in str(row.get("subreddit", "")).lower()),
                int("gaming" in str(row.get("subreddit", "")).lower()),
                int("meme" in str(row.get("subreddit", "")).lower()),
                # è§„åˆ™ç±»å�‹ç‰¹å¾�ï¼ˆåŸºäº�å…³é”®è¯�ï¼‰
                int(
                    any(word in rule for word in ["spam", "advertisement", "promotion"])
                ),
                int(any(word in rule for word in ["harassment", "abuse", "threat"])),
                int(
                    any(
                        word in rule for word in ["off-topic", "relevant", "discussion"]
                    )
                ),
                int(any(word in rule for word in ["personal", "information", "dox"])),
                # è¯„è®ºç±»å�‹ç‰¹å¾�
                int(comment.startswith("http")),  # æ˜¯å�¦ä»¥é“¾æ�¥å¼€å¤´
                int("edit:" in comment or "update:" in comment),  # æ˜¯å�¦åŒ…å�«ç¼–è¾‘
                comment.count("\n"),  # æ�¢è¡Œç¬¦æ•°é‡�
                # æ—¶é—´å’Œæ•°å­—æ¨¡å¼�
                self._count_numbers(comment),
                self._count_dates(comment),
            ]

            features.append(feat)

        return np.array(features)

    def _count_repeated_chars(self, text):
        """è®¡ç®—é‡�å¤�å­—ç¬¦çš„æ•°é‡�"""
        count = 0
        for i in range(len(text) - 1):
            if text[i] == text[i + 1] and text[i].isalpha():
                count += 1
        return count

    def _count_numbers(self, text):
        """è®¡ç®—æ•°å­—çš„æ•°é‡�"""
        import re

        numbers = re.findall(r"\d+", text)
        return len(numbers)

    def _count_dates(self, text):
        """è®¡ç®—æ—¥æœŸæ¨¡å¼�çš„æ•°é‡�"""
        import re

        # ç®€å�•çš„æ—¥æœŸæ¨¡å¼�åŒ¹é…�
        date_patterns = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # MM/DD/YYYY
            r"\d{1,2}-\d{1,2}-\d{2,4}",  # MM-DD-YYYY
            r"\d{4}-\d{1,2}-\d{1,2}",  # YYYY-MM-DD
        ]

        count = 0
        for pattern in date_patterns:
            count += len(re.findall(pattern, text))
        return count


# ==============================================================================
# 5. æœºå™¨å­¦ä¹ æ¨¡å�‹é›†æˆ�
# ==============================================================================
class MLModelEnsemble:
    """
    æœºå™¨å­¦ä¹ æ¨¡å�‹é›†æˆ�
    åŒ…å�«XGBoostã€�LightGBMã€�RandomForestå’ŒLogisticRegression
    """

    def __init__(self):
        self.models = {
            "xgb": xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="auc",
                verbosity=0,
            ),
            "lgb": lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                metric="auc",
                verbose=-1,
            ),
            "rf": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "lr": LogisticRegression(
                C=1.0, random_state=42, max_iter=1000, solver="liblinear"
            ),
        }

    def fit(self, X, y):
        """è®­ç»ƒæ‰€æœ‰æ¨¡å�‹"""
        for name, model in self.models.items():
            print(f"    ğŸ”§ Training {name.upper()}...")
            model.fit(X, y)

    def predict_proba(self, X):
        """é›†æˆ�é¢„æµ‹"""
        predictions = []
        weights = {"xgb": 0.3, "lgb": 0.3, "rf": 0.25, "lr": 0.15}

        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(weights[name] * pred)

        return np.sum(predictions, axis=0)

    def get_feature_importance(self):
        """è�·å�–ç‰¹å¾�é‡�è¦�æ€§"""
        importance_dict = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance_dict[name] = np.abs(model.coef_[0])

        return importance_dict


# ==============================================================================
# 6. æ¨¡å�‹è��å�ˆç­–ç•¥
# ==============================================================================
def smart_blend(dl_preds, ml_preds):
    """
    æ™ºèƒ½è��å�ˆç­–ç•¥
    åŸºäº�é¢„æµ‹ç½®ä¿¡åº¦çš„åŠ¨æ€�æ�ƒé‡�åˆ†é…�
    """
    # åŸºäº�é¢„æµ‹ç½®ä¿¡åº¦çš„åŠ¨æ€�æ�ƒé‡�
    dl_confidence = np.abs(dl_preds - 0.5) * 2  # 0-1ä¹‹é—´ï¼Œè¶Šæ�¥è¿‘0.5ç½®ä¿¡åº¦è¶Šä½�
    ml_confidence = np.abs(ml_preds - 0.5) * 2

    # å½’ä¸€åŒ–æ�ƒé‡�
    total_confidence = dl_confidence + ml_confidence + 1e-8
    dl_weight = dl_confidence / total_confidence
    ml_weight = ml_confidence / total_confidence

    # åŸºç¡€æ�ƒé‡�ï¼šæ·±åº¦å­¦ä¹ æ¨¡å�‹æ�ƒé‡�ç¨�é«˜
    base_dl_weight = 0.6
    base_ml_weight = 0.4

    # æœ€ç»ˆæ�ƒé‡�ï¼šç»“å�ˆåŸºç¡€æ�ƒé‡�å’Œç½®ä¿¡åº¦æ�ƒé‡�
    final_dl_weight = 0.7 * base_dl_weight + 0.3 * dl_weight
    final_ml_weight = 0.7 * base_ml_weight + 0.3 * ml_weight

    # å½’ä¸€åŒ–
    total_weight = final_dl_weight + final_ml_weight
    final_dl_weight /= total_weight
    final_ml_weight /= total_weight

    return final_dl_weight * dl_preds + final_ml_weight * ml_preds


def rank_blend(dl_preds, ml_preds, alpha=0.6):
    """
    åŸºäº�æ�’å��çš„è��å�ˆç­–ç•¥
    """
    from scipy.stats import rankdata

    # å°†é¢„æµ‹å€¼è½¬æ�¢ä¸ºæ�’å��
    dl_ranks = rankdata(dl_preds, method="ordinal")
    ml_ranks = rankdata(ml_preds, method="ordinal")

    # å½’ä¸€åŒ–æ�’å��åˆ°[0,1]
    dl_ranks_norm = (dl_ranks - 1) / (len(dl_ranks) - 1)
    ml_ranks_norm = (ml_ranks - 1) / (len(ml_ranks) - 1)

    # åŠ æ�ƒè��å�ˆ
    blended_ranks = alpha * dl_ranks_norm + (1 - alpha) * ml_ranks_norm

    return blended_ranks


# ==============================================================================
# 7. ä¸»å‡½æ•° - å®Œæ•´è®­ç»ƒå’Œé¢„æµ‹æµ�ç¨‹
# ==============================================================================
def main():
    """
    ä¸»å‡½æ•°ï¼šå®Œæ•´çš„è®­ç»ƒå’Œé¢„æµ‹æµ�ç¨‹
    åŒ…å�«æ·±åº¦å­¦ä¹ æ¨¡å�‹è®­ç»ƒã€�æœºå™¨å­¦ä¹ æ¨¡å�‹è®­ç»ƒå’Œå¤šç§�è��å�ˆç­–ç•¥
    """
    print("=" * 80)
    print("ğŸš€ ULTIMATE ENSEMBLE: Structured Cascade + Advanced ML Models")
    print("=" * 80)

    print("\nğŸ“Š Loading and preprocessing data...")

    # åŠ è½½æ•°æ�®
    train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

    # æ•°æ�®é¢„å¤„ç�†
    for df in [train_df, test_df]:
        for col in [
            "body",
            "rule",
            "positive_example_1",
            "positive_example_2",
            "negative_example_1",
            "negative_example_2",
            "subreddit",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("")

    print(f"    âœ“ Train samples: {len(train_df)}")
    print(f"    âœ“ Test samples: {len(test_df)}")
    print(f"    âœ“ Violation rate: {train_df['rule_violation'].mean():.3f}")

    # ========================================================================
    # é˜¶æ®µ1ï¼šæ·±åº¦å­¦ä¹ æ¨¡å�‹è®­ç»ƒ
    # ========================================================================
    print("\n" + "=" * 60)
    print("ğŸ¤– STAGE 1: Enhanced Structured Cascade Model Training")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    dl_predictions = []

    batch_size = 2
    gradient_accumulation_steps = 16

    print(f"ğŸ“‹ DL Configuration:")
    print(f"    â€¢ Model: microsoft/deberta-v3-large")
    print(f"    â€¢ Batch size: {batch_size}")
    print(f"    â€¢ Gradient accumulation: {gradient_accumulation_steps}")
    print(f"    â€¢ Effective batch size: {batch_size * gradient_accumulation_steps}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # åˆ›å»ºæ–°æ¨¡å�‹å®�ä¾‹
    model = EnhancedStructuredCascadeModel()
    trainer = EnhancedCascadeTrainer(
        model, gradient_accumulation_steps=gradient_accumulation_steps
        )

    best_auc = 0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["rule_violation"])
    ):
        print(f"\nğŸ”„ DL Fold {fold + 1}/3")
        print("-" * 40)

        # å‡†å¤‡æ•°æ�®
        train_ds = StructuredPromptDataset(train_df.iloc[train_idx], tokenizer)
        val_ds = StructuredPromptDataset(train_df.iloc[val_idx], tokenizer)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

        # è®­ç»ƒå¾ªç�¯
        patience = 0
        fold_best_auc = 0

        for epoch in range(8):
            print(f"\n  ğŸ“ˆ Epoch {epoch+1}/8:")

            train_loss, train_auc = trainer.train_epoch(train_loader)
            val_auc = trainer.evaluate(val_loader)

            print(f"    ğŸ�‹ï¸�  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"    ğŸ“Š Val AUC: {val_auc:.4f}")

            # æ¨¡å�‹ä¿�å­˜å’Œæ—©å�œ
            if val_auc > fold_best_auc:
                fold_best_auc = val_auc
                patience = 0
                current_best_state = model.state_dict().copy()
                torch.save(current_best_state, f"model_fold_{fold}.pt")
                print(f"    âœ… New fold best: {fold_best_auc:.4f}")

                if fold_best_auc > best_auc:
                    best_auc = fold_best_auc
                    best_state = current_best_state.copy()
                    torch.save(best_state, "model_global_best.pt")
                    print(f"    ğŸ�† New GLOBAL best: {best_auc:.4f}")
            else:
                patience += 1
                if patience >= 2:
                    print("    â�¹ï¸�  Early stopping")
                    break

        # åŠ è½½æœ€ä½³æ¨¡å�‹è¿›è¡Œé¢„æµ‹
        model.load_state_dict(current_best_state)

        # æµ‹è¯•é›†é¢„æµ‹
        test_ds = StructuredPromptDataset(test_df, tokenizer, mode="test")
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

        row_ids, fold_preds = trainer.predict(test_loader)
        fold_df = pd.DataFrame({"row_id": row_ids, "preds": fold_preds})
        fold_df = fold_df.set_index("row_id").loc[test_df["row_id"]]
        dl_predictions.append(fold_df["preds"].values)

        print(f"  âœ… DL Fold {fold + 1} completed. Best AUC: {fold_best_auc:.4f}")

        # # å†…å­˜æ¸…ç�†
        # del (
        #     model,
        #     trainer,
        #     train_ds,
        #     val_ds,
        #     train_loader,
        #     val_loader,
        #     test_ds,
        #     test_loader,
        # )
        # torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========================================================================
    # é˜¶æ®µ2ï¼šæœºå™¨å­¦ä¹ æ¨¡å�‹è®­ç»ƒ
    # ========================================================================
    print("\n" + "=" * 60)
    print("ğŸ”§ STAGE 2: Advanced Machine Learning Models Training")
    print("=" * 60)

    # ç‰¹å¾�æ��å�–
    print("\nğŸ“Š Feature Engineering:")
    feature_extractor = AdvancedMLFeatureExtractor()

    # æ��å�–è®­ç»ƒç‰¹å¾�
    train_features = feature_extractor.fit_transform(train_df)
    test_features = feature_extractor.transform(test_df)

    print(f"    âœ… Feature extraction completed")
    print(f"    ğŸ“� Feature shape: {train_features.shape}")
    print(
        f"    ğŸ“Š Feature density: {np.count_nonzero(train_features) / train_features.size:.3f}"
    )

    # æœºå™¨å­¦ä¹ æ¨¡å�‹è®­ç»ƒ
    ml_predictions = []
    feature_importances = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["rule_violation"])
    ):
        print(f"\nğŸ”„ ML Fold {fold + 1}/3")
        print("-" * 40)

        X_train = train_features[train_idx]
        X_val = train_features[val_idx]
        y_train = train_df.iloc[train_idx]["rule_violation"].values
        y_val = train_df.iloc[val_idx]["rule_violation"].values

        # è®­ç»ƒMLé›†æˆ�æ¨¡å�‹
        ml_ensemble = MLModelEnsemble()
        ml_ensemble.fit(X_train, y_train)

        # éªŒè¯�
        val_preds = ml_ensemble.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_preds)
        print(f"    ğŸ“Š ML Validation AUC: {val_auc:.4f}")

        # è�·å�–ç‰¹å¾�é‡�è¦�æ€§
        fold_importance = ml_ensemble.get_feature_importance()
        feature_importances.append(fold_importance)

        # æµ‹è¯•é›†é¢„æµ‹
        test_preds = ml_ensemble.predict_proba(test_features)
        ml_predictions.append(test_preds)

        print(f"  âœ… ML Fold {fold + 1} completed. AUC: {val_auc:.4f}")

    # ========================================================================
    # é˜¶æ®µ3ï¼šæ¨¡å�‹è��å�ˆå’Œæœ€ç»ˆé¢„æµ‹
    # ========================================================================
    print("\n" + "=" * 60)
    print("ğŸ”„ STAGE 3: Model Ensemble and Final Predictions")
    print("=" * 60)

    # è®¡ç®—å�„æ¨¡å�‹çš„å¹³å�‡é¢„æµ‹
    print("\nğŸ“Š Computing model predictions...")
    dl_final = np.mean(dl_predictions, axis=0)
    ml_final = np.mean(ml_predictions, axis=0)

    print(f"    ğŸ¤– DL Model - Mean: {dl_final.mean():.4f}, Std: {dl_final.std():.4f}")
    print(f"    ğŸ”§ ML Model - Mean: {ml_final.mean():.4f}, Std: {ml_final.std():.4f}")

    # å¤šç§�è��å�ˆç­–ç•¥
    print("\nğŸ�¯ Applying fusion strategies...")
    fusion_strategies = {
        "dl_only": dl_final,
        "ml_only": ml_final,
        "simple_avg": 0.5 * dl_final + 0.5 * ml_final,
        "dl_weighted": 0.7 * dl_final + 0.3 * ml_final,
        "ml_weighted": 0.3 * dl_final + 0.7 * ml_final,
        "smart_blend": smart_blend(dl_final, ml_final),
        "rank_blend": rank_blend(dl_final, ml_final, alpha=0.6),
    }

    # ========================================================================
    # é˜¶æ®µ4ï¼šç»“æ�œä¿�å­˜å’Œåˆ†æ��
    # ========================================================================
    print("\n" + "=" * 60)
    print("ğŸ’¾ STAGE 4: Saving Results and Analysis")
    print("=" * 60)

    print("\nğŸ“� Saving submission files...")

    for strategy_name, predictions in fusion_strategies.items():
        submission = pd.DataFrame(
            {
                "row_id": test_df["row_id"],
                "rule_violation": np.clip(predictions, 0.001, 0.999),
            }
        )
        submission.to_csv(f"submission_{strategy_name}.csv", index=False)
        print(
            f"    âœ… {strategy_name}: Mean={predictions.mean():.4f}, Std={predictions.std():.4f}"
        )

    # ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��
    print("\nğŸ“ˆ Feature importance analysis...")
    if feature_importances:
        avg_importance = {}
        for model_name in feature_importances[0].keys():
            avg_importance[model_name] = np.mean(
                [fi[model_name] for fi in feature_importances], axis=0
            )
            top_indices = np.argsort(avg_importance[model_name])[-10:][::-1]
            print(
                f"    ğŸ”§ {model_name.upper()} top features: {top_indices[:5].tolist()}"
            )

    # æœ€ç»ˆç»Ÿè®¡
    print("\n" + "=" * 60)
    print("âœ… TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(f"\nğŸ“Š Final Statistics:")
    print(f"    ğŸ�† Best DL AUC achieved: {best_auc:.4f}")
    print(f"    ğŸ¤– DL predictions range: [{dl_final.min():.4f}, {dl_final.max():.4f}]")
    print(f"    ğŸ”§ ML predictions range: [{ml_final.min():.4f}, {ml_final.max():.4f}]")

    print(f"\nğŸ“� Generated files:")
    for strategy in fusion_strategies.keys():
        print(f"    â€¢ submission_{strategy}.csv")

    print(f"\nğŸ�¯ Recommended submission: submission_smart_blend.csv")
    print(f"    (Combines both models with dynamic confidence-based weighting)")


# ==============================================================================
# 8. ç¨‹åº�å…¥å�£
# ==============================================================================
if __name__ == "__main__":
    main()



# æ•²å®šæ–¹æ¡ˆ
import os

# å�Ÿå§‹æ–‡ä»¶å��å’Œæ–°æ–‡ä»¶å��
old_name = 'submission_smart_blend.csv'
new_name = 'submission.csv'

# é‡�å‘½å��æ–‡ä»¶
os.rename(old_name, new_name)

print(f"æ–‡ä»¶å·²ä»� {old_name} é‡�å‘½å��ä¸º {new_name}")

