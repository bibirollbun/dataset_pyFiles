#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
å¤§æ•°æ�®é›†æ™ºèƒ½é‡‡æ ·å™¨
ä»�50GBçš„æ��åº¦ä¸�å¹³è¡¡æ•°æ�®ä¸­è�·å�–ä»£è¡¨æ€§å°�æ ·æœ¬
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
import random

class BigDataSampler:
    """
    å¤§æ•°æ�®é›†é‡‡æ ·å™¨ï¼Œé€‚ç”¨äº�æ��åº¦ä¸�å¹³è¡¡çš„åˆ†å­�-è›‹ç™½è´¨æ•°æ�®
    """
    
    def __init__(self, chunk_size=10000):
        """
        åˆ�å§‹åŒ–é‡‡æ ·å™¨
        
        Args:
            chunk_size: æ¯�æ¬¡è¯»å�–çš„æ•°æ�®å�—å¤§å°�
        """
        self.chunk_size = chunk_size
        self.pos_samples = []
        self.neg_samples = []
        self.protein_stats = {}
        
    def strategy_1_random_sampling(self, file_path, sample_size=50000, pos_ratio=0.1):
        """
        ç­–ç•¥1: åˆ†å±‚éš�æœºé‡‡æ ·
        ä¿�æŒ�å�Ÿå§‹æ•°æ�®çš„ä¸�å¹³è¡¡æ¯”ä¾‹ï¼Œä½†é™�åˆ¶æ ·æœ¬æ€»æ•°
        
        Args:
            file_path: å¤§æ•°æ�®æ–‡ä»¶è·¯å¾„
            sample_size: ç›®æ ‡æ ·æœ¬æ€»æ•°
            pos_ratio: æ­£æ ·æœ¬æ¯”ä¾‹ï¼ˆä¼°è®¡å€¼ï¼‰
        """
        print("ğŸ�¯ ç­–ç•¥1: åˆ†å±‚éš�æœºé‡‡æ ·")
        print(f"   ç›®æ ‡æ ·æœ¬æ•°: {sample_size}")
        print(f"   é¢„ä¼°æ­£æ ·æœ¬æ¯”ä¾‹: {pos_ratio}")
        
        target_pos = int(sample_size * pos_ratio)
        target_neg = sample_size - target_pos
        
        print("ğŸ“– å¼€å§‹è¯»å�–å¤§æ•°æ�®æ–‡ä»¶...")
        
        pos_collected = 0
        neg_collected = 0
        total_processed = 0
        
        # åˆ†å�—è¯»å�–å¤§æ–‡ä»¶
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            total_processed += len(chunk)
            
            # åˆ†ç¦»æ­£è´Ÿæ ·æœ¬
            pos_chunk = chunk[chunk['binds'] == 1]
            neg_chunk = chunk[chunk['binds'] == 0]
            
            # éš�æœºé‡‡æ ·æ­£æ ·æœ¬
            if len(pos_chunk) > 0 and pos_collected < target_pos:
                needed_pos = min(len(pos_chunk), target_pos - pos_collected)
                sampled_pos = pos_chunk.sample(n=needed_pos, random_state=42)
                self.pos_samples.append(sampled_pos)
                pos_collected += needed_pos
            
            # éš�æœºé‡‡æ ·è´Ÿæ ·æœ¬
            if len(neg_chunk) > 0 and neg_collected < target_neg:
                needed_neg = min(len(neg_chunk), target_neg - neg_collected)
                sampled_neg = neg_chunk.sample(n=needed_neg, random_state=42)
                self.neg_samples.append(sampled_neg)
                neg_collected += needed_neg
            
            # æ˜¾ç¤ºè¿›åº¦
            if total_processed % 100000 == 0:
                print(f"   å·²å¤„ç�†: {total_processed:,} è¡Œï¼Œæ­£æ ·æœ¬: {pos_collected}, è´Ÿæ ·æœ¬: {neg_collected}")
            
            # æ£€æŸ¥æ˜¯å�¦å·²è¾¾åˆ°ç›®æ ‡
            if pos_collected >= target_pos and neg_collected >= target_neg:
                break
        
        # å�ˆå¹¶æ ·æœ¬
        if self.pos_samples:
            pos_df = pd.concat(self.pos_samples, ignore_index=True)
        else:
            pos_df = pd.DataFrame()
            
        if self.neg_samples:
            neg_df = pd.concat(self.neg_samples, ignore_index=True)
        else:
            neg_df = pd.DataFrame()
        
        sample_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"âœ… é‡‡æ ·å®Œæˆ�!")
        print(f"   æœ€ç»ˆæ ·æœ¬æ•°: {len(sample_df)}")
        print(f"   æ­£æ ·æœ¬: {len(pos_df)}, è´Ÿæ ·æœ¬: {len(neg_df)}")
        print(f"   å®�é™…æ­£æ ·æœ¬æ¯”ä¾‹: {len(pos_df)/len(sample_df):.4f}")
        
        return sample_df
    
    def strategy_2_balanced_sampling(self, file_path, pos_samples=10000, neg_samples=10000):
        """
        ç­–ç•¥2: å¹³è¡¡é‡‡æ ·
        å¼ºåˆ¶å¹³è¡¡æ­£è´Ÿæ ·æœ¬æ•°é‡�
        
        Args:
            file_path: å¤§æ•°æ�®æ–‡ä»¶è·¯å¾„
            pos_samples: æ­£æ ·æœ¬ç›®æ ‡æ•°é‡�
            neg_samples: è´Ÿæ ·æœ¬ç›®æ ‡æ•°é‡�
        """
        print("âš–ï¸� ç­–ç•¥2: å¹³è¡¡é‡‡æ ·")
        print(f"   ç›®æ ‡æ­£æ ·æœ¬: {pos_samples}")
        print(f"   ç›®æ ‡è´Ÿæ ·æœ¬: {neg_samples}")
        
        pos_collected = 0
        neg_collected = 0
        total_processed = 0
        
        print("ğŸ“– å¼€å§‹è¯»å�–å¤§æ•°æ�®æ–‡ä»¶...")
        
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            total_processed += len(chunk)
            
            # åˆ†ç¦»æ­£è´Ÿæ ·æœ¬
            pos_chunk = chunk[chunk['binds'] == 1]
            neg_chunk = chunk[chunk['binds'] == 0]
            
            # æ”¶é›†æ­£æ ·æœ¬
            if len(pos_chunk) > 0 and pos_collected < pos_samples:
                needed = min(len(pos_chunk), pos_samples - pos_collected)
                self.pos_samples.append(pos_chunk.sample(n=needed, random_state=42))
                pos_collected += needed
            
            # æ”¶é›†è´Ÿæ ·æœ¬
            if len(neg_chunk) > 0 and neg_collected < neg_samples:
                needed = min(len(neg_chunk), neg_samples - neg_collected)
                self.neg_samples.append(neg_chunk.sample(n=needed, random_state=42))
                neg_collected += needed
            
            # æ˜¾ç¤ºè¿›åº¦
            if total_processed % 100000 == 0:
                print(f"   å·²å¤„ç�†: {total_processed:,} è¡Œï¼Œæ­£æ ·æœ¬: {pos_collected}, è´Ÿæ ·æœ¬: {neg_collected}")
            
            # æ£€æŸ¥æ˜¯å�¦å®Œæˆ�
            if pos_collected >= pos_samples and neg_collected >= neg_samples:
                break
        
        # å�ˆå¹¶å¹¶æ‰“ä¹±
        pos_df = pd.concat(self.pos_samples, ignore_index=True) if self.pos_samples else pd.DataFrame()
        neg_df = pd.concat(self.neg_samples, ignore_index=True) if self.neg_samples else pd.DataFrame()
        sample_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"âœ… å¹³è¡¡é‡‡æ ·å®Œæˆ�!")
        print(f"   æœ€ç»ˆæ ·æœ¬æ•°: {len(sample_df)}")
        print(f"   æ­£æ ·æœ¬: {len(pos_df)}, è´Ÿæ ·æœ¬: {len(neg_df)}")
        
        return sample_df
    
    def strategy_3_protein_stratified(self, file_path, samples_per_protein=5000):
        """
        ç­–ç•¥3: æŒ‰è›‹ç™½è´¨åˆ†å±‚é‡‡æ ·
        ç¡®ä¿�æ¯�ç§�è›‹ç™½è´¨éƒ½æœ‰ä»£è¡¨æ€§æ ·æœ¬
        
        Args:
            file_path: å¤§æ•°æ�®æ–‡ä»¶è·¯å¾„
            samples_per_protein: æ¯�ç§�è›‹ç™½è´¨çš„æ ·æœ¬æ•°
        """
        print("ğŸ§ª ç­–ç•¥3: æŒ‰è›‹ç™½è´¨åˆ†å±‚é‡‡æ ·")
        print(f"   æ¯�ç§�è›‹ç™½è´¨ç›®æ ‡æ ·æœ¬æ•°: {samples_per_protein}")
        
        protein_samples = {}
        total_processed = 0
        
        print("ğŸ“– å¼€å§‹è¯»å�–å¤§æ•°æ�®æ–‡ä»¶...")
        
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            total_processed += len(chunk)
            
            # æŒ‰è›‹ç™½è´¨åˆ†ç»„
            for protein in chunk['protein_name'].unique():
                if protein not in protein_samples:
                    protein_samples[protein] = {'pos': [], 'neg': []}
                
                protein_chunk = chunk[chunk['protein_name'] == protein]
                pos_chunk = protein_chunk[protein_chunk['binds'] == 1]
                neg_chunk = protein_chunk[protein_chunk['binds'] == 0]
                
                # æ¯�ç§�è›‹ç™½è´¨æ”¶é›†ä¸€å®šæ•°é‡�çš„æ­£è´Ÿæ ·æœ¬
                target_per_class = samples_per_protein // 2
                
                if len(pos_chunk) > 0 and len(protein_samples[protein]['pos']) < target_per_class:
                    needed = min(len(pos_chunk), target_per_class - len(protein_samples[protein]['pos']))
                    protein_samples[protein]['pos'].extend(pos_chunk.sample(n=needed, random_state=42).to_dict('records'))
                
                if len(neg_chunk) > 0 and len(protein_samples[protein]['neg']) < target_per_class:
                    needed = min(len(neg_chunk), target_per_class - len(protein_samples[protein]['neg']))
                    protein_samples[protein]['neg'].extend(neg_chunk.sample(n=needed, random_state=42).to_dict('records'))
            
            # æ˜¾ç¤ºè¿›åº¦
            if total_processed % 100000 == 0:
                progress = {p: len(v['pos']) + len(v['neg']) for p, v in protein_samples.items()}
                print(f"   å·²å¤„ç�†: {total_processed:,} è¡Œï¼Œå�„è›‹ç™½è´¨æ ·æœ¬æ•°: {progress}")
            
            # æ£€æŸ¥æ˜¯å�¦æ‰€æœ‰è›‹ç™½è´¨éƒ½æ”¶é›†å¤Ÿäº†
            if all(len(v['pos']) + len(v['neg']) >= samples_per_protein for v in protein_samples.values()):
                break
        
        # å�ˆå¹¶æ‰€æœ‰è›‹ç™½è´¨æ ·æœ¬
        all_samples = []
        for protein, samples in protein_samples.items():
            all_samples.extend(samples['pos'])
            all_samples.extend(samples['neg'])
        
        sample_df = pd.DataFrame(all_samples).sample(frac=1, random_state=42)
        
        print(f"âœ… åˆ†å±‚é‡‡æ ·å®Œæˆ�!")
        print(f"   é‡‡æ ·çš„è›‹ç™½è´¨æ•°: {len(protein_samples)}")
        print(f"   æœ€ç»ˆæ ·æœ¬æ•°: {len(sample_df)}")
        for protein, samples in protein_samples.items():
            pos_count = len(samples['pos'])
            neg_count = len(samples['neg'])
            print(f"   {protein}: æ­£æ ·æœ¬ {pos_count}, è´Ÿæ ·æœ¬ {neg_count}")
        
        return sample_df
    
    def estimate_data_distribution(self, file_path, max_rows=100000):
        """
        ä¼°è®¡å¤§æ•°æ�®é›†çš„åˆ†å¸ƒæƒ…å†µ
        """
        print("ğŸ”� ä¼°è®¡æ•°æ�®åˆ†å¸ƒ...")
        
        total_rows = 0
        pos_count = 0
        protein_counts = Counter()
        
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            if total_rows >= max_rows:
                break
                
            total_rows += len(chunk)
            pos_count += (chunk['binds'] == 1).sum()
            
            for protein in chunk['protein_name']:
                protein_counts[protein] += 1
        
        pos_ratio = pos_count / total_rows
        
        print(f"   é‡‡æ ·è¡Œæ•°: {total_rows:,}")
        print(f"   æ­£æ ·æœ¬æ¯”ä¾‹: {pos_ratio:.4f}")
        print(f"   è›‹ç™½è´¨åˆ†å¸ƒ: {dict(protein_counts)}")
        
        return pos_ratio, protein_counts

def main():
    """
    ä¸»å‡½æ•°ï¼šæ¼”ç¤ºä¸�å�Œé‡‡æ ·ç­–ç•¥
    """
    print("ğŸ�¯ å¤§æ•°æ�®é›†é‡‡æ ·å™¨")
    print("=" * 50)
    
    # é…�ç½®
    big_data_file = "/kaggle/input/leash-BELKA/train.csv"  # æ›¿æ�¢ä¸ºä½ çš„50GBæ–‡ä»¶è·¯å¾„
    
    # æ£€æŸ¥æ–‡ä»¶æ˜¯å�¦å­˜åœ¨
    if not os.path.exists(big_data_file):
        print(f"â�Œ æ–‡ä»¶ä¸�å­˜åœ¨: {big_data_file}")
        print("è¯·ä¿®æ”¹ big_data_file å�˜é‡�ä¸ºä½ çš„å®�é™…æ–‡ä»¶è·¯å¾„")
        
        # æ¼”ç¤ºç”¨ï¼šä½¿ç”¨ç�°æœ‰çš„balanced_train.csv
        print("\nğŸ“� æ¼”ç¤ºæ¨¡å¼�ï¼šä½¿ç”¨ balanced_train.csv")
        big_data_file = "balanced_train.csv"
    
    sampler = BigDataSampler(chunk_size=5000)
    
    # ä¼°è®¡æ•°æ�®åˆ†å¸ƒ
    pos_ratio, protein_counts = sampler.estimate_data_distribution(big_data_file)
    
    print(f"\nğŸ�² é€‰æ‹©é‡‡æ ·ç­–ç•¥:")
    print(f"1. åˆ†å±‚éš�æœºé‡‡æ · (ä¿�æŒ�å�Ÿå§‹ä¸�å¹³è¡¡æ¯”ä¾‹)")
    print(f"2. å¹³è¡¡é‡‡æ · (å¼ºåˆ¶1:1æ¯”ä¾‹)")
    print(f"3. æŒ‰è›‹ç™½è´¨åˆ†å±‚é‡‡æ · (æ¯�ç§�è›‹ç™½è´¨å�‡åŒ€)")
    
    strategy = input("è¯·é€‰æ‹©ç­–ç•¥ (1/2/3): ")
    
    if strategy == "1":
        sample_df = sampler.strategy_1_random_sampling(
            big_data_file, 
            sample_size=50000, 
            pos_ratio=pos_ratio
        )
        output_file = "sampled_data_random.csv"
    elif strategy == "2":
        sample_df = sampler.strategy_2_balanced_sampling(
            big_data_file,
            pos_samples=20000,
            neg_samples=20000
        )
        output_file = "sampled_data_balanced.csv"
    elif strategy == "3":
        sample_df = sampler.strategy_3_protein_stratified(
            big_data_file,
            samples_per_protein=20000
        )
        output_file = "sampled_data_stratified.csv"
    else:
        print("â�Œ æ— æ•ˆé€‰æ‹©")
        return
    
    # ä¿�å­˜é‡‡æ ·ç»“æ�œ
    sample_df.to_csv(output_file, index=False)
    print(f"\nğŸ’¾ é‡‡æ ·æ•°æ�®å·²ä¿�å­˜: {output_file}")
    
    # æ˜¾ç¤ºé‡‡æ ·ç»“æ�œç»Ÿè®¡
    print(f"\nğŸ“Š é‡‡æ ·ç»“æ�œç»Ÿè®¡:")
    print(f"   æ€»æ ·æœ¬æ•°: {len(sample_df)}")
    print(f"   æ­£æ ·æœ¬æ•°: {(sample_df['binds'] == 1).sum()}")
    print(f"   è´Ÿæ ·æœ¬æ•°: {(sample_df['binds'] == 0).sum()}")
    print(f"   æ­£æ ·æœ¬æ¯”ä¾‹: {sample_df['binds'].mean():.4f}")
    print(f"   è›‹ç™½è´¨åˆ†å¸ƒ:")
    for protein, count in sample_df['protein_name'].value_counts().items():
        print(f"     {protein}: {count}")

if __name__ == "__main__":
    main()


pip install rdkit


# ğŸ› ï¸� ç¬¬ä¸€æ­¥ï¼šè¶…çº§å¼ºåŠ›æŠ‘åˆ¶è­¦å‘Šå’Œå¯¼å…¥åº“
import warnings
import sys
import os

# è¶…çº§é�™éŸ³æ¨¡å¼� - å¤šé‡�ä¿�é™©
warnings.filterwarnings('ignore')  # æŠ‘åˆ¶æ‰€æœ‰è­¦å‘Š
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')
warnings.filterwarnings('ignore', message='.*DEPRECATION WARNING.*')

# è®¾ç½®ç�¯å¢ƒå�˜é‡�
os.environ['RDKIT_SILENCE_WARNING'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

# ç¦�ç”¨RDKitæ—¥å¿—
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# è®¾ç½®matplotlibä¸­æ–‡å­—ä½“
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("âœ… åº“å¯¼å…¥å®Œæˆ�ï¼Œè¶…çº§é�™éŸ³æ¨¡å¼�å·²å�¯ç”¨")



def modern_smiles_to_fingerprint(smiles, n_bits=1024):
    """
    ç�°ä»£åŒ–SMILESåˆ°åˆ†å­�æŒ‡çº¹è½¬æ�¢ï¼Œé€‚ç”¨äº�Jupyter
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    
    # ä½¿ç”¨æ–°çš„MorganGenerator APIï¼ˆæ— è­¦å‘Šï¼‰
    try:
        from rdkit.Chem import rdMolDescriptors
        generator = rdMolDescriptors.GetMorganGenerator(radius=2)
        fp = generator.GetFingerprint(mol, nBits=n_bits)
    except (ImportError, AttributeError):
        # é�™é»˜å›�é€€åˆ°æ—§API
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    
    return np.array(fp)




# ğŸ“– ç¬¬ä¸‰æ­¥ï¼šè¯»å�–æ•°æ�®
print("ğŸ“– æ­£åœ¨è¯»å�–æ•°æ�®...")
df = pd.read_csv('sampled_data_stratified.csv')

print(f"æ•°æ�®å½¢çŠ¶: {df.shape}")
print(f"è›‹ç™½è´¨ç§�ç±»: {df['protein_name'].unique()}")
print(f"ç»“å�ˆæ¯”ä¾‹: {df['binds'].mean():.3f}")

# æ˜¾ç¤ºå‰�å‡ è¡Œ
df.head()
    


def train_svm_model(X, y):
    """
    è®­ç»ƒSVMæ¨¡å�‹
    """
    print("\nğŸš„ å¼€å§‹è®­ç»ƒSVMæ¨¡å�‹...")
    
    # åˆ’åˆ†æ•°æ�®
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ç‰¹å¾�æ ‡å‡†åŒ–
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # è®¡ç®—ç±»åˆ«æ�ƒé‡�ï¼ˆå¤„ç�†ä¸�å¹³è¡¡æ•°æ�®ï¼‰
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    class_weight = {0: 1, 1: n_neg/n_pos} if n_pos > 0 else None
    
    print(f"   - è®­ç»ƒé›†: {len(y_train)} æ ·æœ¬")
    print(f"   - æµ‹è¯•é›†: {len(y_test)} æ ·æœ¬")
    print(f"   - æ­£è´Ÿæ ·æœ¬æ¯”ä¾‹: 1:{n_neg/n_pos:.2f}")
    print(f"   - ç±»åˆ«æ�ƒé‡�: {class_weight}")


# ğŸ”¬ ç¬¬å››æ­¥ï¼šç‰¹å¾�å‡†å¤‡ï¼ˆä½¿ç”¨å°�æ ·æœ¬ï¼Œé€‚å�ˆJupyterå¿«é€Ÿæµ‹è¯•ï¼‰
print("ğŸ”¬ å‡†å¤‡ç‰¹å¾�...")

# ä½¿ç”¨è¾ƒå°�æ ·æœ¬è¿›è¡Œå¿«é€Ÿæ¼”ç¤º
sample_size = 5000  # åœ¨Jupyterä¸­ä½¿ç”¨è¾ƒå°�æ ·æœ¬
df_sample = df.sample(n=sample_size, random_state=42)

print(f"ä½¿ç”¨æ ·æœ¬æ•°: {len(df_sample)}")

# æ��å�–åˆ†å­�æŒ‡çº¹ï¼ˆä½¿ç”¨æ— è­¦å‘Šç‰ˆæœ¬ï¼‰
print("æ­£åœ¨æ��å�–åˆ†å­�æŒ‡çº¹ï¼ˆæ— è­¦å‘Šç‰ˆæœ¬ï¼‰...")
fingerprints = []
for i, smiles in enumerate(df_sample['molecule_smiles']):
    fp = modern_smiles_to_fingerprint(smiles, n_bits=512)  # ä½¿ç”¨æ— è­¦å‘Šå‡½æ•°
    fingerprints.append(fp)
    
    # æ˜¾ç¤ºè¿›åº¦
    if (i + 1) % 5000 == 0:
        print(f"å·²å¤„ç�†: {i + 1}/{len(df_sample)}")

fingerprints = np.array(fingerprints)
print(f"åˆ†å­�æŒ‡çº¹å½¢çŠ¶: {fingerprints.shape}")

# è›‹ç™½è´¨ç¼–ç �
le = LabelEncoder()
protein_encoded = le.fit_transform(df_sample['protein_name'])
protein_onehot = pd.get_dummies(protein_encoded, prefix='protein')

# å�ˆå¹¶ç‰¹å¾�
X = np.hstack([fingerprints, protein_onehot.values])
y = df_sample['binds'].values

print(f"æœ€ç»ˆç‰¹å¾�å½¢çŠ¶: {X.shape}")
print(f"æ ‡ç­¾åˆ†å¸ƒ: {np.bincount(y)}")
print("âœ… ç‰¹å¾�å‡†å¤‡å®Œæˆ�ï¼Œåº”è¯¥æ²¡æœ‰çœ‹åˆ°ä»»ä½•è­¦å‘Šï¼�")


# ğŸš„ ç¬¬äº”æ­¥ï¼šè®­ç»ƒSVMæ¨¡å�‹
print("ğŸš„ è®­ç»ƒSVMæ¨¡å�‹...")

# åˆ’åˆ†æ•°æ�®
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# æ ‡å‡†åŒ–
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# è®¡ç®—ç±»åˆ«æ�ƒé‡�
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
class_weight = {0: 1, 1: n_neg/n_pos} if n_pos > 0 else None

print(f"è®­ç»ƒé›†: {len(y_train)}, æµ‹è¯•é›†: {len(y_test)}")
print(f"ç±»åˆ«æ�ƒé‡�: {class_weight}")

# è®­ç»ƒSVM
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    class_weight=class_weight,
    random_state=42
)

svm.fit(X_train_scaled, y_train)
print("âœ… æ¨¡å�‹è®­ç»ƒå®Œæˆ�")



# ğŸ“Š ç¬¬å…­æ­¥ï¼šæ¨¡å�‹è¯„ä¼°
print("ğŸ“Š æ¨¡å�‹è¯„ä¼°...")

# é¢„æµ‹
y_pred = svm.predict(X_test_scaled)
y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]

# è®¡ç®—æŒ‡æ ‡
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {auc_score:.4f}")

# åˆ†ç±»æŠ¥å‘Š
print("\nåˆ†ç±»æŠ¥å‘Š:")
print(classification_report(y_test, y_pred))

# æ··æ·†çŸ©é˜µå�¯è§†åŒ–
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Binding', 'Binding'],
            yticklabels=['No Binding', 'Binding'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print(f"æœ€ç»ˆAUCå¾—åˆ†: {auc_score:.4f}")


# ğŸ”® å¯¹test.csvè¿›è¡Œé¢„æµ‹ (è¾“å‡ºç«�èµ›æ ¼å¼�)
def predict_new_data(test_file='/kaggle/input/leash-BELKA/test.csv', trained_model=svm, trained_scaler=scaler, 
                     trained_encoder=le, output_file='submission1.csv'):
    """
    ä½¿ç”¨è®­ç»ƒå¥½çš„æ¨¡å�‹å¯¹æ–°æ•°æ�®è¿›è¡Œé¢„æµ‹
    æœ€ç»ˆè¾“å‡ºå�ªåŒ…å�« id å’Œ binds ä¸¤åˆ—
    """
    print(f"ğŸ“– è¯»å�–æµ‹è¯•æ•°æ�®: {test_file}")
    
    try:
        # è¯»å�–æµ‹è¯•æ•°æ�®
        df_test = pd.read_csv(test_file)
        print(f"æµ‹è¯•æ•°æ�®å½¢çŠ¶: {df_test.shape}")
        print(f"åˆ—å��: {list(df_test.columns)}")
        
        # æ£€æŸ¥å¿…è¦�åˆ—
        required_cols = ['molecule_smiles', 'protein_name']
        if not all(col in df_test.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_test.columns]
            print(f"â�Œ ç¼ºå°‘å¿…è¦�åˆ—: {missing}")
            return None
        
        # æ£€æŸ¥æ˜¯å�¦æœ‰idåˆ—
        if 'id' not in df_test.columns:
            print("âš ï¸�  æ²¡æœ‰å�‘ç�°idåˆ—ï¼Œå°†ä½¿ç”¨ç´¢å¼•ä½œä¸ºid")
            df_test['id'] = df_test.index
        
        # æ��å�–ç‰¹å¾�ï¼ˆä½¿ç”¨ç›¸å�Œçš„æ–¹æ³•ï¼‰
        print("ğŸ§¬ æ��å�–åˆ†å­�æŒ‡çº¹...")
        test_fingerprints = []
        for i, smiles in enumerate(df_test['molecule_smiles']):
            fp = web_safe_smiles_to_fingerprint(smiles, n_bits=512)  # ä¸�è®­ç»ƒæ—¶ä¿�æŒ�ä¸€è‡´
            test_fingerprints.append(fp)
            
            if (i + 1) % 10000 == 0:  # å¤§æ•°æ�®é›†ï¼Œæ¯�1ä¸‡æ�¡æ˜¾ç¤ºè¿›åº¦
                print(f"å·²å¤„ç�†: {i + 1}/{len(df_test)}")
        
        test_fingerprints = np.array(test_fingerprints)
        
        # è›‹ç™½è´¨ç¼–ç �
        print("ğŸ§ª ç¼–ç �è›‹ç™½è´¨...")
        try:
            test_protein_encoded = trained_encoder.transform(df_test['protein_name'])
        except ValueError as e:
            print(f"å�‘ç�°æ–°è›‹ç™½è´¨ï¼Œæ— æ³•é¢„æµ‹: {e}")
            return None
        
        test_protein_onehot = pd.get_dummies(test_protein_encoded, prefix='protein')
        
        # ç¡®ä¿�ç‰¹å¾�ç»´åº¦ä¸€è‡´
        if test_protein_onehot.shape[1] != protein_onehot.shape[1]:
            print("âš ï¸�  æµ‹è¯•æ•°æ�®è›‹ç™½è´¨ç¼–ç �ç»´åº¦ä¸�åŒ¹é…�ï¼Œè¿›è¡Œè°ƒæ•´...")
            # è¡¥é½�ç¼ºå¤±çš„åˆ—
            for col in protein_onehot.columns:
                if col not in test_protein_onehot.columns:
                    test_protein_onehot[col] = 0
            # é‡�æ–°æ�’åº�åˆ—
            test_protein_onehot = test_protein_onehot[protein_onehot.columns]
        
        # å�ˆå¹¶ç‰¹å¾�
        X_test = np.hstack([test_fingerprints, test_protein_onehot.values])
        
        # æ ‡å‡†åŒ–ï¼ˆä½¿ç”¨è®­ç»ƒæ—¶çš„scalerï¼‰
        print("âš–ï¸� æ ‡å‡†åŒ–ç‰¹å¾�...")
        X_test_scaled = trained_scaler.transform(X_test)
        
        # é¢„æµ‹
        print("ğŸ�¯ è¿›è¡Œé¢„æµ‹...")
        y_pred = trained_model.predict(X_test_scaled)
        y_pred_proba = trained_model.predict_proba(X_test_scaled)[:, 1]
        
        # ğŸ�† å‡†å¤‡ç«�èµ›æ��äº¤æ ¼å¼�ï¼šå�ªä¿�ç•™ id å’Œ binds
        submission_df = pd.DataFrame({
            'id': df_test['id'],
            'binds': y_pred  # ç›´æ�¥ä½¿ç”¨é¢„æµ‹ç»“æ�œ (0æˆ–1)
        })
        
        # ä¿�å­˜ç«�èµ›æ ¼å¼�ç»“æ�œ
        submission_df.to_csv(output_file, index=False)
        
        # ğŸ“Š æ˜¾ç¤ºç»Ÿè®¡ä¿¡æ�¯
        print(f"\nğŸ“Š é¢„æµ‹ç»“æ�œç»Ÿè®¡:")
        print(f"æ€»æ ·æœ¬æ•°: {len(submission_df)}")
        print(f"é¢„æµ‹ç»“å�ˆ: {np.sum(y_pred == 1)} ({np.mean(y_pred):.1%})")
        print(f"é¢„æµ‹ä¸�ç»“å�ˆ: {np.sum(y_pred == 0)} ({1-np.mean(y_pred):.1%})")
        print(f"å¹³å�‡ç»“å�ˆæ¦‚ç�‡: {np.mean(y_pred_proba):.3f}")
        
        # å�„è›‹ç™½è´¨ç»Ÿè®¡
        print(f"\nğŸ§ª å�„è›‹ç™½è´¨é¢„æµ‹:")
        for protein in df_test['protein_name'].unique():
            mask = df_test['protein_name'] == protein
            protein_pred = y_pred[mask]
            protein_proba = y_pred_proba[mask]
            print(f"{protein}: {len(protein_pred)} æ ·æœ¬, ç»“å�ˆç�‡: {np.mean(protein_pred):.3f}, å¹³å�‡æ¦‚ç�‡: {np.mean(protein_proba):.3f}")
        
        print(f"\nğŸ’¾ ç«�èµ›æ��äº¤æ–‡ä»¶å·²ä¿�å­˜: {output_file}")
        print(f"ğŸ“‹ æ��äº¤æ ¼å¼�: id, binds")
        
        # æ˜¾ç¤ºæ��äº¤æ–‡ä»¶çš„å‰�å‡ è¡Œ
        print(f"\nğŸ”� æ��äº¤æ–‡ä»¶é¢„è§ˆ:")
        print(submission_df.head(10))
        
        print(f"\nğŸ“ˆ æ��äº¤æ–‡ä»¶ç»Ÿè®¡:")
        print(f"æ–‡ä»¶å¤§å°�: {len(submission_df)} è¡Œ")
        print(f"åˆ—å��: {list(submission_df.columns)}")
        print(f"bindså€¼åˆ†å¸ƒ: {submission_df['binds'].value_counts().to_dict()}")
        
        return submission_df
        
    except FileNotFoundError:
        print(f"â�Œ æœªæ‰¾åˆ°æ–‡ä»¶: {test_file}")
        print("è¯·ç¡®ä¿�æµ‹è¯•æ–‡ä»¶è·¯å¾„æ­£ç¡®")
        return None
    except Exception as e:
        print(f"â�Œ é¢„æµ‹è¿‡ç¨‹å‡ºé”™: {e}")
        import traceback
        traceback.print_exc()
        return None

# æ‰§è¡Œé¢„æµ‹
print("ğŸ�¯ å¼€å§‹é¢„æµ‹test.csv (ç”Ÿæˆ�ç«�èµ›æ��äº¤æ ¼å¼�)...")
submission_results = predict_new_data()

if submission_results is not None:
    print("âœ… é¢„æµ‹å®Œæˆ�ï¼�")
    print("ğŸ�† ç«�èµ›æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�ï¼Œæ ¼å¼�: id, binds")
    # æ˜¾ç¤ºç»“æ�œçš„å‰�å‡ è¡Œ
    display(submission_results.head())


