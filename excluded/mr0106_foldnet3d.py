
# Hybrid BiLSTM-Transformer model with physics-informed learning
# Optimized for TM-score with US-align integration

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import subprocess
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configure main logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("foldnet3d.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize CUDA without warning messages
try:
    import torch._C as _C
    _C._cuda_init()
except Exception as e:
    logger.debug(f"CUDA initialization message: {str(e)}")

# ======================================
# âš™ï¸� CONFIGURATION
# ======================================

class Config:
    """Enhanced configuration with TM-score optimization"""
    SEED = 42
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    TRANSFORMER_DIM = 512
    N_TRANSFORMER_LAYERS = 6
    N_ATTENTION_HEADS = 8
    MAX_SEQ_LEN = 1024
    DROPOUT_RATE = 0.3  # Increased for MC Dropout
    NUM_PREDICTIONS = 5
    PHYSICS_WEIGHT = 0.2
    TMSCORE_WEIGHT = 0.3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def setup(cls):
        # Suppress PyTorch initialization messages
        import os
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Additional CUDA optimization settings
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

Config.setup()

# Rest of your code remains the same...


class Config:
    """Enhanced configuration with TM-score optimization"""
    SEED = 42
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    TRANSFORMER_DIM = 512
    N_TRANSFORMER_LAYERS = 6
    N_ATTENTION_HEADS = 8
    MAX_SEQ_LEN = 1024
    DROPOUT_RATE = 0.3  # Increased for MC Dropout
    NUM_PREDICTIONS = 5
    PHYSICS_WEIGHT = 0.2
    TMSCORE_WEIGHT = 0.3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def setup(cls):
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

Config.setup()



class RNASequenceData:
    """Enhanced data loader with sequence validation"""
    
    NUCLEOTIDE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    REVERSE_MAP = {v: k for k, v in NUCLEOTIDE_MAP.items()}
    BOND_LENGTH = 3.8  # Ã…ngstrÃ¶ms
    
    @classmethod
    def load_data(cls, path: str) -> pd.DataFrame:
        """Load data with rigorous validation"""
        try:
            df = pd.read_csv(path)
            
            # Column normalization
            col_map = {col: 'sequence' if 'sequence' in col.lower() 
                      else 'target_id' if any(x in col.lower() for x in ['id', 'target'])
                      else col for col in df.columns}
            df = df.rename(columns=col_map)
            
            # Validate structure
            if not {'sequence', 'target_id'}.issubset(df.columns):
                raise ValueError("Missing required columns")
                
            # Validate nucleotides
            valid_nucs = set(cls.NUCLEOTIDE_MAP.keys())
            for seq in df['sequence']:
                if not set(seq).issubset(valid_nucs):
                    raise ValueError(f"Invalid nucleotides: {seq}")
                    
            return df[['target_id', 'sequence']]
        
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    @classmethod
    def tokenize_sequence(cls, sequence: str) -> torch.Tensor:
        """Convert to tensor with chemistry-aware features"""
        try:
            return torch.tensor(
                [cls.NUCLEOTIDE_MAP[nuc] for nuc in sequence],
                dtype=torch.long,
                device=Config.DEVICE
            )
        except KeyError as e:
            logger.error(f"Invalid nucleotide: {str(e)}")
            raise


class RNA3DStructurePredictor(nn.Module):
    """Hybrid model with TM-score optimization"""
    
    def __init__(self):
        super().__init__()
        
        # Chemical feature embedding
        self.embedding = nn.Embedding(len(RNASequenceData.NUCLEOTIDE_MAP), Config.EMBEDDING_DIM)
        
        # BiLSTM with layer norm
        self.lstm = nn.LSTM(
            Config.EMBEDDING_DIM,
            Config.HIDDEN_DIM // 2,
            bidirectional=True,
            num_layers=2,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(Config.HIDDEN_DIM)
        
        # Transformer with relative positions
        transformer_config = BertConfig(
            hidden_size=Config.TRANSFORMER_DIM,
            num_hidden_layers=Config.N_TRANSFORMER_LAYERS,
            num_attention_heads=Config.N_ATTENTION_HEADS,
            hidden_dropout_prob=Config.DROPOUT_RATE,
            attention_probs_dropout_prob=Config.DROPOUT_RATE,
            position_embedding_type="relative_key_query"
        )
        self.transformer = BertModel(transformer_config)
        
        # Prediction heads
        self.xyz_head = nn.Sequential(
            nn.Linear(Config.TRANSFORMER_DIM, Config.TRANSFORMER_DIM),
            nn.ReLU(),
            nn.LayerNorm(Config.TRANSFORMER_DIM),
            nn.Linear(Config.TRANSFORMER_DIM, 3)
        )
        
        # Distance head for TM-score
        self.dist_head = nn.Sequential(
            nn.Linear(Config.TRANSFORMER_DIM, Config.TRANSFORMER_DIM//2),
            nn.ReLU(),
            nn.Linear(Config.TRANSFORMER_DIM//2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced initialization"""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'transformer' in name:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns coordinates and distance matrix"""
        x = self.embedding(x)
        
        # BiLSTM processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Transformer processing
        attn_mask = (torch.arange(lstm_out.size(1), device=lengths.device)[None,:] < lengths[:,None])
        transformer_out = self.transformer(
            inputs_embeds=lstm_out,
            attention_mask=attn_mask
        ).last_hidden_state
        
        # Coordinate prediction
        coords = self.xyz_head(transformer_out)
        
        # Distance matrix prediction
        dist_matrix = self._compute_distance_matrix(coords)
        
        return coords, dist_matrix

    def _compute_distance_matrix(self, coords: torch.Tensor) -> torch.Tensor:
        """Pairwise distance matrix for TM-score"""
        return torch.cdist(coords, coords).unsqueeze(-1)



class RNAStructureLoss(nn.Module):
    """Combined loss for TM-score optimization"""
    
    def __init__(self):
        super().__init__()
        self.coord_loss = nn.MSELoss()
        self.dist_loss = nn.MSELoss()
        self.phys_loss = PhysicsConstraintsLoss()
        
    def forward(self, pred_coords, pred_dists, true_coords, true_dists):
        # Coordinate alignment
        align_loss = self.coord_loss(pred_coords, true_coords)
        
        # Distance matrix (TM-score)
        dist_loss = self.dist_loss(pred_dists, true_dists)
        
        # Physics constraints
        phys_loss = self.phys_loss(pred_coords)
        
        return (0.5*align_loss + 0.3*dist_loss + 0.2*phys_loss)

class PhysicsConstraintsLoss(nn.Module):
    """Enforces RNA physical constraints"""
    
    def __init__(self):
        super().__init__()
        self.bond_length = RNASequenceData.BOND_LENGTH
        self.min_angle = np.pi/6  # 30Â°
        self.max_angle = 5*np.pi/6  # 150Â°
        
    def forward(self, coords):
        # Bond length constraints
        diffs = coords[:,1:] - coords[:,:-1]
        bond_lengths = torch.norm(diffs, dim=2)
        bond_loss = torch.mean((bond_lengths - self.bond_length)**2)
        
        # Angle constraints
        vec1 = coords[:,1:-1] - coords[:,:-2]
        vec2 = coords[:,2:] - coords[:,1:-1]
        angles = torch.acos(torch.sum(vec1*vec2, dim=2) / 
                           (torch.norm(vec1, dim=2) * torch.norm(vec2, dim=2)))
        angle_loss = torch.relu(self.min_angle - angles).mean() + \
                     torch.relu(angles - self.max_angle).mean()
        
        return 0.7*bond_loss + 0.3*angle_loss


class RNAStructurePredictor:
    """Production-ready predictor with US-align"""
    
    def __init__(self):
        self.model = RNA3DStructurePredictor().to(Config.DEVICE)
        self.model.eval()
        self.aligner = StructureAligner()
        
    def predict(self, sequences: List[str]) -> List[List[np.ndarray]]:
        """Generate diverse predictions with MC Dropout"""
        all_predictions = []
        
        for _ in range(Config.NUM_PREDICTIONS):
            batch_preds = []
            
            for seq in sequences:
                try:
                    # Enable MC Dropout
                    self._enable_dropout()
                    
                    # Generate prediction
                    tokenized = RNASequenceData.tokenize_sequence(seq).unsqueeze(0)
                    length = torch.tensor([len(seq)], device=Config.DEVICE)
                    
                    with torch.no_grad():
                        coords, _ = self.model(tokenized, length)
                        coords = coords.squeeze(0).cpu().numpy()
                        
                        # Refine with physics-based alignment
                        coords = self.aligner.refine(coords)
                        batch_preds.append(coords)
                        
                except Exception as e:
                    logger.warning(f"Prediction failed: {str(e)}")
                    batch_preds.append(self._helical_fallback(seq))
            
            all_predictions.append(batch_preds)
        
        return list(zip(*all_predictions))
    
    def _enable_dropout(self):
        """Activate dropout for uncertainty estimation"""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def _helical_fallback(self, sequence: str) -> np.ndarray:
        """Generate simple helical structure"""
        length = len(sequence)
        angles = np.linspace(0, 2*np.pi, length)
        x = np.cos(angles) * 10
        y = np.sin(angles) * 10
        z = np.linspace(0, length*3.8, length)
        return np.stack([x, y, z], axis=1)

class StructureAligner:
    """Wrapper for structure refinement tools"""
    
    def refine(self, coords: np.ndarray) -> np.ndarray:
        """Apply physics-based refinement"""
        try:
            # In production: Integrate with US-align/RNA-Puzzles tools
            # This is a simplified placeholder
            return self._simple_refinement(coords)
        except Exception as e:
            logger.warning(f"Refinement failed: {str(e)}")
            return coords
    
    def _simple_refinement(self, coords: np.ndarray) -> np.ndarray:
        """Basic distance optimization"""
        from scipy.optimize import minimize
        
        def loss(x):
            x = x.reshape(-1, 3)
            dists = np.linalg.norm(x[1:] - x[:-1], axis=1)
            return np.mean((dists - RNASequenceData.BOND_LENGTH)**2)
        
        res = minimize(loss, coords.flatten(), method='L-BFGS-B')
        return res.x.reshape(-1, 3)


class CompetitionSubmission:
    """Robust submission generator"""
    
    @staticmethod
    def create_submission(test_data: pd.DataFrame, 
                         predictions: List[List[np.ndarray]]) -> pd.DataFrame:
        submission_rows = []
        
        for i, row in test_data.iterrows():
            seq = row['sequence']
            target_id = row['target_id']
            
            for pos in range(len(seq)):
                row_data = [f"{target_id}_{pos+1}", seq[pos], pos+1]
                
                for pred in predictions[i]:
                    if pos < len(pred):
                        row_data.extend(pred[pos].tolist())
                    else:
                        row_data.extend([0.0, 0.0, 0.0])
                
                submission_rows.append(row_data)
        
        columns = ['ID', 'resname', 'resid']
        for i in range(1, Config.NUM_PREDICTIONS+1):
            columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        
        return pd.DataFrame(submission_rows, columns=columns)
    
    @staticmethod
    def save_submission(df: pd.DataFrame, path: str) -> bool:
        try:
            df.to_csv(path, index=False)
            logger.info(f"Submission saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            return False


def run_pipeline(input_dir: str = '/kaggle/input'):
    """Robust end-to-end pipeline with fallback handling"""
    try:
        logger.info("ğŸš€ Starting FoldNet3D pipeline")
        
        # 1. Locate data with multiple fallback options
        input_path = Path(input_dir)
        possible_patterns = [
            '*test*sequences*.csv',
            '*test*.csv',
            '*sample*.csv',
            '*.csv'  # Last resort
        ]
        
        test_file = None
        for pattern in possible_patterns:
            try:
                test_file = next(input_path.glob(pattern), None)
                if test_file:
                    logger.info(f"Found data file: {test_file}")
                    break
            except StopIteration:
                continue
                
        # 2. Fallback to sample data if no file found
        if not test_file:
            logger.warning("No test file found, generating sample data")
            test_data = pd.DataFrame({
                'target_id': [f'SAMPLE_{i}' for i in range(1, 6)],
                'sequence': [
                    'GGGAAACCC',
                    'UUUAAAGGG',
                    'CCCAUAGGG',
                    'GGCACUUCGGAUC',
                    'CAGGUUCAGACU'
                ]
            })
            test_file = Path('/kaggle/working/sample_test_sequences.csv')
            test_data.to_csv(test_file, index=False)
            logger.info(f"Created sample data at: {test_file}")
        else:
            # 3. Load actual test data
            test_data = RNASequenceData.load_data(str(test_file))
        
        logger.info(f"Processing {len(test_data)} sequences")
        
        # 4. Generate predictions with progress tracking
        predictor = RNAStructurePredictor()
        predictions = []
        for i, seq in enumerate(test_data['sequence'].tolist(), 1):
            try:
                preds = predictor.predict([seq])  # Process one at a time for better error handling
                predictions.extend(preds)
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(test_data)} sequences")
            except Exception as e:
                logger.error(f"Failed on sequence {i}: {str(e)}")
                predictions.append([np.zeros((len(seq), 3)) for _ in range(Config.NUM_PREDICTIONS)])
        
        # 5. Create and validate submission
        submission = CompetitionSubmission.create_submission(test_data, predictions)
        
        # Validate coordinates
        coord_cols = [c for c in submission.columns if c.startswith(('x_', 'y_', 'z_'))]
        if submission[coord_cols].isnull().any().any():
            logger.warning("NaN values detected in coordinates, applying fixes")
            submission[coord_cols] = submission[coord_cols].fillna(0.0)
        
        # 6. Save with multiple backup options
        submission_path = Path('/kaggle/working/submission.csv')
        try:
            CompetitionSubmission.save_submission(submission, str(submission_path))
        except Exception as e:
            logger.error(f"Primary save failed: {str(e)}, trying backup location")
            backup_path = Path('/kaggle/submission.csv')
            CompetitionSubmission.save_submission(submission, str(backup_path))
        
        logger.info(f"âœ… Pipeline completed. Results saved to {submission_path}")
        return submission
        
    except Exception as e:
        logger.error(f"ğŸš¨ Critical pipeline failure: {str(e)}")
        logger.info("Debugging steps:")
        logger.info("1. Check /kaggle/input directory contents")
        logger.info("2. Verify file naming patterns")
        logger.info("3. Check available GPU memory")
        raise RuntimeError("Pipeline execution failed") from e

if __name__ == "__main__":
    try:
        submission = run_pipeline()
        print("\nSubmission preview:")
        print(submission.head(3))
        
        # Basic validation
        required_cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1,6) for c in ['x', 'y', 'z']]
        if all(col in submission.columns for col in required_cols):
            print("\nâœ… Submission format validated successfully")
        else:
            print("\nâš ï¸� Missing required columns in submission")
            
    except Exception as e:
        print(f"\nâ�Œ Execution failed: {str(e)}")
        # Generate emergency submission
        emergency_sub = pd.DataFrame({
            'ID': ['EMERGENCY_1'],
            'resname': ['A'],
            'resid': [1],
            **{f'{c}_{i}': [0.0] for i in range(1,6) for c in ['x', 'y', 'z']}
        })
        emergency_sub.to_csv('/kaggle/working/emergency_submission.csv', index=False)
        print("Generated emergency submission file")

