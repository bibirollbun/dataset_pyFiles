import numpy as np
import pandas as pd
from tqdm import tqdm
import os


file_paths = {
    'train_sequences_v1': '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv',
    'train_sequences_v2': '/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv',
    'train_labels_v1': '/kaggle/input/stanford-rna-3d-folding/train_labels.csv',
    'train_labels_v2': '/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv', 
    'validation_sequences': '/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv',
    'validation_labels': '/kaggle/input/stanford-rna-3d-folding/validation_labels.csv',
    'test_sequences': '/kaggle/input/stanford-rna-3d-folding/test_sequences.csv',
    'msa_folder': '/kaggle/input/stanford-rna-3d-folding/MSA/'
}

for path in file_paths.keys():
    if os.path.exists(os.path.join(os.getcwd(), file_paths[path])):
        train_sequences = pd.concat([pd.read_csv(file_paths['train_sequences_v1']), pd.read_csv(file_paths['train_sequences_v2'])], axis=0)
        train_labels = pd.concat([pd.read_csv(file_paths['train_labels_v1']), pd.read_csv(file_paths['train_labels_v2'])], axis=0)
        validation_sequences = pd.read_csv(file_paths['validation_sequences'])
        validation_labels = pd.read_csv(file_paths['validation_labels'])
        test_sequences = pd.read_csv(file_paths['test_sequences'])
        MSA_FOLDER = file_paths['msa_folder']
    else:
        pass



MAX_SEQ_LENGTH = 10


print("max sequence: ", MAX_SEQ_LENGTH)

class FastaAlignment:  
    class SeqRecord:
        def __init__(self, seq, id="", description=""):
            self.seq = seq
            self.id = id
            self.description = description
    
    def __init__(self):
        self.records = []
    
    def append(self, record):
        self.records.append(record)
    
    def get_alignment_length(self):
        if not self.records:
            return 0
        return len(self.records[0].seq)
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        return self.records[index]

def read_fasta(file_path):
    alignment = FastaAlignment()
    
    try:
        with open(file_path, 'r') as fasta_file:
            current_id = ""
            current_description = ""
            current_seq = ""
            
            for line in fasta_file:
                line = line.strip()
                
                if not line: 
                    continue
                    
                if line.startswith('>'):  
                    if current_seq:
                        record = FastaAlignment.SeqRecord(current_seq, current_id, current_description)
                        alignment.append(record)
                    
                    header_parts = line[1:].split(maxsplit=1)
                    current_id = header_parts[0]
                    current_description = header_parts[1] if len(header_parts) > 1 else ""
                    current_seq = ""
                else:  
                    current_seq += line
            
            if current_seq:
                record = FastaAlignment.SeqRecord(current_seq, current_id, current_description)
                alignment.append(record)
    except:
        return None
    
    if len(alignment) > 0:
        first_len = len(alignment[0].seq)
        for record in alignment.records:
            if len(record.seq) != first_len:
                print(f"Warning: Sequences in {file_path} have different lengths.")
    
    return alignment if len(alignment) > 0 else None

class RNAData:
    def __init__(self, input_sequences, input_labels):
        try:
            self.sequence = [row[-1]['sequence'] for row in input_sequences.iterrows()]
            self.target_id = [row[-1]['target_id'] for row in input_sequences.iterrows()]
        except:
            self.sequence = []
            self.target_id = []
        
        self.coords_dict = {}
        
        if input_labels is not None:
            try:
                for target_id in self.target_id:
                    matching_rows = input_labels[input_labels['ID'].str.startswith(target_id)]
                    if not matching_rows.empty:
                        coords = matching_rows[['x_1', 'y_1', 'z_1']].values
                        self.coords_dict[target_id] = coords
            except:
                pass

    def one_hot_encode(self, sequence):
        nucleotide_map = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
        
        valid_nucleotides = [nucleotide for nucleotide in sequence if nucleotide in nucleotide_map]
        if not valid_nucleotides:
            return np.zeros((1, 4))
        encoding = [nucleotide_map[nucleotide] for nucleotide in sequence if nucleotide in nucleotide_map]
        return np.array(encoding)

    def filter_extreme_coordinates(self):
        removed_targets = []
        for target_id in list(self.coords_dict.keys()):
            coords = self.coords_dict[target_id]
            if np.max(np.abs(coords)) > 1e10:
                print(f"Filtering out target {target_id} with extreme coordinate values")
                removed_targets.append(target_id)
                del self.coords_dict[target_id]
        
        if removed_targets:
            indices_to_keep = [i for i, tid in enumerate(self.target_id) if tid not in removed_targets]
            self.target_id = [self.target_id[i] for i in indices_to_keep]
            self.sequence = [self.sequence[i] for i in indices_to_keep]
            print(f"Removed {len(removed_targets)} targets with extreme coordinate values")
        
        return removed_targets

    def truncate_sequence(self, sequence, coords, max_length):
        if len(sequence) <= max_length:
            return sequence, coords
        
        truncated_seq = sequence[:max_length]
        
        if coords is not None and len(coords) > 0:
            truncated_coords = coords[:max_length] if len(coords) > max_length else coords
        else:
            truncated_coords = coords
            
        return truncated_seq, truncated_coords

class Geodata(RNAData):
    def __init__(self, input_sequences, input_labels):
        super(Geodata, self).__init__(input_sequences, input_labels)
        if input_labels is not None:
            try:
                coords_array = input_labels[['x_1', 'y_1', 'z_1']].to_numpy()
                valid_mask = np.isfinite(coords_array).all(axis=1)
                coords_array = coords_array[valid_mask]
                finite_mask = np.all(np.abs(coords_array) <= 1e10, axis=1)
                coords_array = coords_array[finite_mask]
                
                mean = np.mean(coords_array, axis=0)
                std = np.std(coords_array, axis=0)

                self.normalization_params = {'mean': mean, 'std': std}
            except:
                self.normalization_params = {'mean': np.zeros(3), 'std': np.ones(3)}
        else:
            self.normalization_params = {'mean': np.zeros(3), 'std': np.ones(3)}

    def predict_secondary_structure_simple(self, sequence):
        seq_length = len(sequence)
        dot_bracket = '.' * seq_length
        pairing_matrix = np.zeros((seq_length, seq_length))
        bpp_matrix = np.zeros((seq_length, seq_length))
        
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        pairs = []
        for i in range(seq_length - 4): 
            for j in range(i + 4, seq_length):
                if sequence[i] in complement and sequence[j] == complement[sequence[i]]:
                    if sequence[i] in ['G', 'C']:
                        score = 0.8
                    else:
                        score = 0.6  
                    
                    distance_factor = max(0.1, 1.0 - (j - i) / seq_length)
                    final_score = score * distance_factor
                    
                    pairs.append((i, j, final_score))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        used_positions = set()
        
        dot_bracket_list = list(dot_bracket)
        for i, j, score in pairs:
            if i not in used_positions and j not in used_positions:
                dot_bracket_list[i] = '('
                dot_bracket_list[j] = ')'
                pairing_matrix[i, j] = 1
                pairing_matrix[j, i] = 1
                bpp_matrix[i, j] = score
                bpp_matrix[j, i] = score
                used_positions.add(i)
                used_positions.add(j)
        
        dot_bracket = ''.join(dot_bracket_list)
        
        return {
            'dot_bracket': dot_bracket,
            'mfe': -len([c for c in dot_bracket if c in '()']) * 2.0,  
            'pairing_matrix': pairing_matrix,
            'bpp_matrix': bpp_matrix
        }
    
    def calculate_position_specific_features(self, sequence, sec_struct_data):
        seq_length = len(sequence)
        features = np.zeros((seq_length, 10))
        for i in range(seq_length):
            window_start = max(0, i-2)
            window_end = min(seq_length, i+3)
            window = sequence[window_start:window_end]
            
            features[i, 0] = window.count('A') / len(window)
            features[i, 1] = window.count('C') / len(window)
            features[i, 2] = window.count('G') / len(window)
            features[i, 3] = window.count('U') / len(window)
            
            features[i, 4] = 1 if sec_struct_data['dot_bracket'][i] == '(' else 0
            features[i, 5] = 1 if sec_struct_data['dot_bracket'][i] == ')' else 0
            features[i, 6] = 1 if sec_struct_data['dot_bracket'][i] == '.' else 0
            
            features[i, 7] = np.sum(sec_struct_data['bpp_matrix'][i]) if hasattr(sec_struct_data['bpp_matrix'], 'shape') else 0
            features[i, 8] = i / seq_length
            features[i, 9] = (seq_length - i) / seq_length
        
        return features
        
    def augment_geo_data(self, coords):
        if coords.shape[0] == 0:
            return coords
            
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        if np.max(np.abs(coords)) > 1e10:
            return coords
        
        try:
            theta_x = np.random.uniform(0, 2*np.pi)
            theta_y = np.random.uniform(0, 2*np.pi)
            theta_z = np.random.uniform(0, 2*np.pi)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            Ry = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])
                        
            Rz = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ])
            
            R = np.dot(Rz, np.dot(Ry, Rx))
            rotated_coords = coords.copy()
            
            if coords.shape[1] >= 3:
                valid_mask = ~np.all(coords == 0, axis=1)
                
                if np.any(valid_mask):
                    center = np.mean(coords[valid_mask], axis=0)
                    centered_coords = coords[valid_mask] - center
                    rotated_points = np.dot(centered_coords, R.T)
                    rotated_coords[valid_mask] = rotated_points + center
            
            return rotated_coords
        except:
            return coords

    def normalize_coordinates(self, coords):
        if coords.shape[0] == 0:
            return coords
        
        if coords.ndim == 1:
            coords = np.expand_dims(coords, axis=0)
        
        try:
            normalized_coords = (coords - self.normalization_params['mean']) / self.normalization_params['std']
            return normalized_coords
        except:
            return coords
            
    def prepare_geo_data(self, augment=False, num_augmentations=0, MAX_SEQ_LENGTH=None):
        if MAX_SEQ_LENGTH is None:
            MAX_SEQ_LENGTH = globals().get('MAX_SEQ_LENGTH', 500)
        
        self.filter_extreme_coordinates()
        X_geo = []
        Y_coords = []
        metadata = []
        
        for target_id, seq in tqdm(zip(self.target_id, self.sequence), desc="Processing geometric data", total=len(self.sequence)):
            coords = self.coords_dict.get(target_id, None)
            
            truncated_seq, truncated_coords = self.truncate_sequence(seq, coords, MAX_SEQ_LENGTH)
            
            predicted_structure = self.predict_secondary_structure_simple(truncated_seq)
            position_features = self.calculate_position_specific_features(truncated_seq, predicted_structure)
            
            if truncated_coords is not None:
                normalized_coords = self.normalize_coordinates(truncated_coords)
                coords_pad = np.zeros((MAX_SEQ_LENGTH, 3))
                valid_len = min(len(normalized_coords), MAX_SEQ_LENGTH)
                coords_pad[:valid_len] = normalized_coords[:valid_len]
                Y_coords.append(coords_pad)
            else:
                coords_pad = np.zeros((MAX_SEQ_LENGTH, 3))
                Y_coords.append(coords_pad)
            
            one_hot = self.one_hot_encode(truncated_seq)
            min_length = min(len(one_hot), len(position_features))
            one_hot = one_hot[:min_length]
            position_features = position_features[:min_length]
            
            if position_features.shape[0] > 0 and one_hot.shape[0] > 0:
                combined_features = np.concatenate((one_hot, position_features), axis=1)
                padded_features = np.zeros((MAX_SEQ_LENGTH, combined_features.shape[1]))
                padded_features[:len(combined_features)] = combined_features
                X_geo.append(padded_features)
                
                metadata.append({
                    'target_id': target_id,
                    'seq_length': len(truncated_seq),
                    'original_seq_length': len(seq),
                    'truncated': len(seq) > MAX_SEQ_LENGTH
                })
                
                if augment and truncated_coords is not None:
                    if len(truncated_coords) > 0 and np.max(np.abs(truncated_coords)) <= 1e10:
                        for aug_idx in range(num_augmentations):
                            augmented_coords = self.augment_geo_data(truncated_coords)
                            normalized_aug_coords = self.normalize_coordinates(augmented_coords)
                            
                            augmented_pad = np.zeros((MAX_SEQ_LENGTH, 3))
                            valid_len = min(len(normalized_aug_coords), MAX_SEQ_LENGTH)
                            augmented_pad[:valid_len] = normalized_aug_coords[:valid_len]
                            
                            X_geo.append(padded_features)  
                            Y_coords.append(augmented_pad)
                            
                            metadata.append({
                                'target_id': target_id,
                                'seq_length': len(truncated_seq),
                                'original_seq_length': len(seq),
                                'truncated': len(seq) > MAX_SEQ_LENGTH,
                                'augmentation': aug_idx + 1
                            })
        
        X_geo_array = np.array(X_geo)
        Y_coords_array = np.array(Y_coords)
        
        if np.isnan(X_geo_array).any():
            print(f"Found {np.isnan(X_geo_array).sum()} NaN values in X_geo. Replacing with zeros.")
            X_geo_array = np.nan_to_num(X_geo_array)
            
        if np.isnan(Y_coords_array).any():
            print(f"Found {np.isnan(Y_coords_array).sum()} NaN values in Y_coords. Replacing with zeros.")
            Y_coords_array = np.nan_to_num(Y_coords_array)
        
        if np.isinf(X_geo_array).any():
            print(f"Found {np.isinf(X_geo_array).sum()} infinity values in X_geo. Replacing with large values.")
            X_geo_array = np.nan_to_num(X_geo_array, posinf=1e6, neginf=-1e6)
            
        if np.isinf(Y_coords_array).any():
            print(f"Found {np.isinf(Y_coords_array).sum()} infinity values in Y_coords. Replacing with large values.")
            Y_coords_array = np.nan_to_num(Y_coords_array, posinf=1e6, neginf=-1e6)
        
        print(f"Generated {len(X_geo_array)} samples from {len(self.sequence)} sequences")
        truncated_count = sum([1 for m in metadata if m.get('truncated', False)])
        print(f"Truncated {truncated_count} sequences that were longer than {MAX_SEQ_LENGTH}")
        
        return X_geo_array, Y_coords_array, self.normalization_params, metadata
    
class MSAData(RNAData):
    def __init__(self, input_sequences, input_labels=None, msa_folder=None, verbose=True, filtered_targets=None):
        super(MSAData, self).__init__(input_sequences, input_labels)
        self.msa_folder = msa_folder
        self.verbose = verbose
        self.msa_data = {}
        
        if filtered_targets is not None:
            indices_to_keep = [i for i, tid in enumerate(self.target_id) if tid not in filtered_targets]
            self.target_id = [self.target_id[i] for i in indices_to_keep]
            self.sequence = [self.sequence[i] for i in indices_to_keep]
            if self.verbose:
                print(f"MSAData: Filtered out {len(filtered_targets)} targets to maintain consistency")
    
    def load_msa_files(self):
        loaded = 0
        skipped = 0
        
        if not self.msa_folder or not os.path.exists(self.msa_folder):
            if self.verbose:
                print(f"MSA folder '{self.msa_folder}' doesn't exist")
            return self.msa_data
            
        for target_id in tqdm(self.target_id, desc="Loading MSA files"):
            if target_id in self.msa_data:  
                pass
                
            possible_paths = [os.path.join(self.msa_folder, f"{target_id}.MSA.fasta")]
            for msa_file_path in possible_paths:
                if os.path.exists(msa_file_path) and os.path.getsize(msa_file_path) > 0:
                    alignment = read_fasta(msa_file_path)
                    if alignment and len(alignment) > 1:
                        self.msa_data[target_id] = alignment
                        loaded += 1
                    else:
                        skipped += 1
        
        if self.verbose:
            print(f"Loaded {loaded} MSA files, skipped {skipped}")
            if loaded == 0:
                print("Warning: No MSA files were loaded. Check the MSA_FOLDER path and file format.")
                
        return self.msa_data
    
    def prepare_msa_data(self, MAX_SEQ_LENGTH=None):
        if MAX_SEQ_LENGTH is None:
            MAX_SEQ_LENGTH = globals().get('MAX_SEQ_LENGTH', 500)
            
        self.load_msa_files() 
        all_msa_features = []
        metadata = []
    
        feature_size = 2 + MAX_SEQ_LENGTH
        
        for target_id, seq in tqdm(zip(self.target_id, self.sequence), desc="Preparing MSA data"):
            truncated_seq, _ = self.truncate_sequence(seq, None, MAX_SEQ_LENGTH)
            
            padded_features = np.zeros((MAX_SEQ_LENGTH, feature_size))
            
            if target_id in self.msa_data:
                try:
                    alignment = self.msa_data[target_id]
                    msa_features = self.process_alignment(
                        alignment, seq, truncated_seq, MAX_SEQ_LENGTH
                    )
                    
                    if msa_features.shape[1] != feature_size:
                        if self.verbose:
                            print(f"Warning: Inconsistent feature size for {target_id}: {msa_features.shape[1]} vs {feature_size}")
                        if msa_features.shape[1] < feature_size:
                            temp = np.zeros((msa_features.shape[0], feature_size))
                            temp[:, :msa_features.shape[1]] = msa_features
                            msa_features = temp
                        else:
                            msa_features = msa_features[:, :feature_size]
                    
                    valid_len = min(msa_features.shape[0], MAX_SEQ_LENGTH)
                    padded_features[:valid_len, :] = msa_features[:valid_len, :]
                except:
                    pass
            
            all_msa_features.append(padded_features)
            
            metadata.append({
                'target_id': target_id,
                'seq_length': len(truncated_seq),
                'original_seq_length': len(seq),
                'truncated': len(seq) > MAX_SEQ_LENGTH
            })
            
        if self.verbose:
            print(f"MSA features shape: {len(all_msa_features)} x {all_msa_features[0].shape[0]} x {all_msa_features[0].shape[1]}")
            print(f"Generated {len(all_msa_features)} MSA samples from {len(self.sequence)} sequences")
            truncated_count = sum([1 for m in metadata if m.get('truncated', False)])
            print(f"Truncated {truncated_count} sequences that were longer than {MAX_SEQ_LENGTH}")
            
        all_msa_features_array = np.array(all_msa_features)
        if np.isnan(all_msa_features_array).any() or np.isinf(all_msa_features_array).any():
            if self.verbose:
                print(f"Found {np.isnan(all_msa_features_array).sum()} NaN and {np.isinf(all_msa_features_array).sum()} infinity values in MSA features.")
            all_msa_features_array = np.nan_to_num(all_msa_features_array)
                
        return all_msa_features_array, metadata
    
    def process_alignment(self, alignment, full_seq, truncated_seq, MAX_SEQ_LENGTH):
        alignment_length = alignment.get_alignment_length()
        num_sequences = len(alignment)
        seq_length = len(truncated_seq)
        
        features = np.zeros((seq_length, 2 + MAX_SEQ_LENGTH))
    
        target_idx = None
        for i, record in enumerate(alignment):
            if full_seq in str(record.seq).replace('-', ''):
                target_idx = i
                
        if target_idx is None:
            return features
        
        seq_to_align_map = {}
        seq_pos = 0
        target_seq = str(alignment[target_idx].seq)
        
        for align_pos, char in enumerate(target_seq):
            if char != '-': 
                if seq_pos < len(full_seq):
                    seq_to_align_map[seq_pos] = align_pos
                    seq_pos += 1
        
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        pssm = np.zeros((4, alignment_length))
        
        columns = []
        for j in range(alignment_length):
            column = [str(rec.seq)[j] for rec in alignment]
            columns.append(column)
            
            for nucleotide in column:
                if nucleotide.upper() in nucleotide_map:
                    idx = nucleotide_map[nucleotide.upper()]
                    pssm[idx, j] += 1
        
        column_sums = np.sum(pssm, axis=0)
        column_sums[column_sums == 0] = 1 
        frequencies = pssm / column_sums[np.newaxis, :]
        
        epsilon = 1e-10
        entropy = -np.sum(frequencies * np.log2(frequencies + epsilon), axis=0)
        max_entropy = -np.log2(0.25)
        conservation_scores = 1 - (entropy / max_entropy)
        conservation_scores = np.nan_to_num(conservation_scores)
        
        coverage = []
        for j in range(alignment_length):
            coverage.append(1.0 - columns[j].count('-') / num_sequences)
        
        truncated_align_positions = []
        for pos in range(seq_length):
            if pos in seq_to_align_map:
                truncated_align_positions.append(seq_to_align_map[pos])
        
        correlation_matrix = {}
        
        for idx_i, align_pos_i in enumerate(truncated_align_positions):
            if align_pos_i >= len(columns):
                continue
            col_i = columns[align_pos_i]
            target_val_i = target_seq[align_pos_i]
            
            for idx_j, align_pos_j in enumerate(truncated_align_positions):
                if idx_i != idx_j and align_pos_j < len(columns):
                    col_j = columns[align_pos_j]
                    target_val_j = target_seq[align_pos_j]
                    
                    matches = 0
                    valid_pairs = 0
                    
                    for val_i, val_j in zip(col_i, col_j):
                        if val_i != '-' and val_j != '-':
                            valid_pairs += 1
                            if (val_i == target_val_i) == (val_j == target_val_j):
                                matches += 1
                    
                    corr = matches / valid_pairs if valid_pairs > 0 else 0
                    correlation_matrix[(idx_i, idx_j)] = corr
        
        for seq_pos in range(seq_length):
            if seq_pos in seq_to_align_map:
                align_pos = seq_to_align_map[seq_pos]
                
                if align_pos < len(conservation_scores):
                    features[seq_pos, 0] = conservation_scores[align_pos]
                
                features[seq_pos, 1] = coverage[align_pos] if align_pos < len(coverage) else 0
                
                if seq_pos < len(truncated_align_positions):
                    for other_seq_pos in range(min(seq_length, MAX_SEQ_LENGTH)):
                        if other_seq_pos in seq_to_align_map and seq_pos != other_seq_pos:
                            if other_seq_pos < len(truncated_align_positions):
                                features[seq_pos, 2 + other_seq_pos] = correlation_matrix.get((seq_pos, other_seq_pos), 0)
        
        return features


LIMIT = 5
AUGMENT = False
NUM_AUG = 5

if train_sequences is not None:
    limited_train_sequences = train_sequences[:LIMIT]
    limited_validation_sequences = validation_sequences[:LIMIT]
    
    geo_train = Geodata(limited_train_sequences, train_labels)
    X_train_geo, Y_train_coords, normalization_params, train_metadata = geo_train.prepare_geo_data()
    
    geo_val = Geodata(limited_validation_sequences, validation_labels)
    X_val_geo, Y_val_coords, _, _ = geo_val.prepare_geo_data()
    
    # filtered_train_targets = set([row[-1]['target_id'] for row in limited_train_sequences.iterrows()]) - set(geo_train.target_id)
    # filtered_val_targets = set([row[-1]['target_id'] for row in limited_validation_sequences.iterrows()]) - set(geo_val.target_id)
    
    #X_train_msa, train_metadata = MSAData(limited_train_sequences, train_labels, MSA_FOLDER, filtered_targets=filtered_train_targets).prepare_msa_data()
    #X_val_msa, _ = MSAData(limited_validation_sequences, validation_labels, MSA_FOLDER, filtered_targets=filtered_val_targets).prepare_msa_data()

print("X_train_geo: ", X_train_geo.shape)
#print("X_train_msa: ", X_train_msa.shape)
print("Y_train: ", Y_train_coords.shape)
print("X_val_geo: ", X_val_geo.shape)
#print("X_val_msa: ", X_val_msa.shape)
print("Y_val: ", Y_val_coords.shape)
print("variance of y train: ", np.var(Y_train_coords))
print("variance of y val: ", np.var(Y_val_coords))


import tensorflow as tf
import keras as keras
from keras.src.saving import load_model
from keras.src.models import Model, Sequential
from keras.src.layers import Input, Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Lambda, Reshape
from keras.src.optimizers import Adam
from keras.src.metrics import Metric
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, TerminateOnNaN


BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 0.001
TRANSFORMER_HEADS = 8
DROPOUT_RATE = 0.2
TRANSFORMER_UNITS = 256
HIDDEN_DIM = 128
FF_DIM = 512


@keras.src.saving.register_keras_serializable()
class TriangularAttention(Layer):
    def __init__(self, hidden_dim=128, num_heads=4, dropout_rate=0.1, **kwargs):
        super(TriangularAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.head_dim, dropout=self.dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)
        
        self.triangle_dense1 = Dense(self.hidden_dim, activation='relu')
        self.triangle_dense2 = Dense(self.hidden_dim)
        
        super(TriangularAttention, self).build(input_shape)
        
    def call(self, inputs, training=False):
        attn_output = self.attention(query=inputs, key=inputs, value=inputs, training=training)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        seq_len = tf.shape(out1)[1]
        
        reshaped = tf.reshape(out1, [-1, seq_len, 1, self.hidden_dim])
        repeated = tf.tile(reshaped, [1, 1, seq_len, 1])
        
        pairwise_features = tf.concat([repeated, tf.transpose(repeated, [0, 2, 1, 3])], axis=-1)
        triangle_out = self.triangle_dense1(pairwise_features)
        triangle_out = self.triangle_dense2(triangle_out)
        triangle_out = tf.reduce_mean(triangle_out, axis=2)
        output = self.layernorm2(out1 + triangle_out)
        return output
    


@keras.saving.register_keras_serializable()
class CrossAttentionBlock(Layer):
    def __init__(self, hidden_dim=128, num_heads=4, dropout_rate=0.1, **kwargs):
        super(CrossAttentionBlock, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # bio to geo
        self.cross_attention_1to2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads, dropout=self.dropout_rate)
        
        # geo to bio
        self.cross_attention_2to1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads, dropout=self.dropout_rate)
        
        self.ffn_1 = Sequential([
            Dense(self.hidden_dim * 4, activation='relu'), 
            Dropout(self.dropout_rate), 
            Dense(self.hidden_dim)
        ])
        
        self.ffn_2 = Sequential([
            Dense(self.hidden_dim * 4, activation='relu'), 
            Dropout(self.dropout_rate), 
            Dense(self.hidden_dim)
        ])
        
        self.layernorm_1a = LayerNormalization(epsilon=1e-6)
        self.layernorm_1b = LayerNormalization(epsilon=1e-6)
        self.layernorm_2a = LayerNormalization(epsilon=1e-6)
        self.layernorm_2b = LayerNormalization(epsilon=1e-6)
        
        self.dropout = Dropout(self.dropout_rate)
        
        super(CrossAttentionBlock, self).build(input_shape)
        
    def call(self, inputs, training=False):
        tower1_input, tower2_input = inputs
        
        attn_1to2_output = self.cross_attention_1to2(query=tower1_input, key=tower2_input, value=tower2_input, training=training)
        attn_1to2_output = self.dropout(attn_1to2_output, training=training)
        tower1_output_temp = self.layernorm_1a(tower1_input + attn_1to2_output)
        
        ffn1_output = self.ffn_1(tower1_output_temp, training=training)
        ffn1_output = self.dropout(ffn1_output, training=training)
        tower1_output = self.layernorm_1b(tower1_output_temp + ffn1_output)
        
        attn_2to1_output = self.cross_attention_2to1(query=tower2_input, key=tower1_input, value=tower1_input, training=training)
        attn_2to1_output = self.dropout(attn_2to1_output, training=training)
        tower2_output_temp = self.layernorm_2a(tower2_input + attn_2to1_output)
    
        ffn2_output = self.ffn_2(tower2_output_temp, training=training)
        ffn2_output = self.dropout(ffn2_output, training=training)
        tower2_output = self.layernorm_2b(tower2_output_temp + ffn2_output)
        
        return tower1_output, tower2_output
    

@keras.saving.register_keras_serializable()
class BiologyTower(Layer):
    def __init__(self, hidden_dim=128, num_layers=3, num_heads=4, dropout_rate=0.1, **kwargs):
        super(BiologyTower, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.input_projection = Dense(self.hidden_dim)
        
        self.transformer_blocks = [
            TriangularAttention(
                hidden_dim=self.hidden_dim, 
                num_heads=self.num_heads, 
                dropout_rate=self.dropout_rate, 
                name=f"biology_triangular_attention_{i}"
            ) for i in range(self.num_layers)
        ]
        
        self.layernorms = [LayerNormalization(epsilon=1e-6) for _ in range(self.num_layers)]
        self.dropouts = [Dropout(self.dropout_rate) for _ in range(self.num_layers)]
        
        self.output_projection = Dense(self.hidden_dim)
        
        super(BiologyTower, self).build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.input_projection(inputs)
    
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, training=training)
            x = self.dropouts[i](x, training=training)
            x = self.layernorms[i](x)
        
        return self.output_projection(x)
    

@keras.saving.register_keras_serializable()
class GeometryTower(Layer):
    def __init__(self, hidden_dim=128, num_layers=3, num_heads=4, dropout_rate=0.1, **kwargs):
        super(GeometryTower, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.input_projection = Dense(self.hidden_dim)

        self.transformer_blocks = [
            TriangularAttention(
                hidden_dim=self.hidden_dim, 
                num_heads=self.num_heads, 
                dropout_rate=self.dropout_rate, 
                name=f"geometry_triangular_attention_{i}"
            ) for i in range(self.num_layers)
        ]

        self.layernorms = [LayerNormalization(epsilon=1e-6) for _ in range(self.num_layers)]
        self.dropouts = [Dropout(self.dropout_rate) for _ in range(self.num_layers)]
        
        self.distance_mlp = Sequential([
            Dense(self.hidden_dim * 2, activation='relu'), 
            Dropout(self.dropout_rate), 
            Dense(self.hidden_dim, activation='relu'), 
            Dense(1)
        ])
        
        self.output_projection = Dense(self.hidden_dim)
        
        super(GeometryTower, self).build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.input_projection(inputs)
        
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, training=training)
            x = self.dropouts[i](x, training=training)
            x = self.layernorms[i](x)
        
        return self.output_projection(x)
    
    
@keras.saving.register_keras_serializable()
class RNAStructurePredictor(Model):
    def __init__(self, max_seq_length, msa_feature_dim, geo_feature_dim, 
                 hidden_dim=128, num_tower_layers=3, num_cross_layers=2, 
                 num_heads=4, dropout_rate=0.1, **kwargs):
        super(RNAStructurePredictor, self).__init__(**kwargs)
        
        self.max_seq_length = max_seq_length
        #self.msa_feature_dim = msa_feature_dim
        self.geo_feature_dim = geo_feature_dim
        self.hidden_dim = hidden_dim
        self.num_tower_layers = num_tower_layers
        self.num_cross_layers = num_cross_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape=None):
        # if input_shape is None:
        #     msa_input_shape = (None, self.max_seq_length, self.msa_feature_dim)
        #     geo_input_shape = (None, self.max_seq_length, self.geo_feature_dim)
        # else:
        #     msa_input_shape, geo_input_shape = input_shape
        geo_input_shape = input_shape
        
        self.biology_tower = BiologyTower(
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_tower_layers, 
            num_heads=self.num_heads, 
            dropout_rate=self.dropout_rate
        )
        
        self.geometry_tower = GeometryTower(
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_tower_layers, 
            num_heads=self.num_heads, 
            dropout_rate=self.dropout_rate
        )
        
        self.cross_attention_blocks = [
            CrossAttentionBlock(
                hidden_dim=self.hidden_dim, 
                num_heads=self.num_heads, 
                dropout_rate=self.dropout_rate, 
                name=f"cross_attention_block_{i}"
            ) for i in range(self.num_cross_layers)
        ]
        
        self.coordinate_prediction = Sequential([
            Dense(self.hidden_dim * 2, activation='relu'), 
            Dropout(self.dropout_rate), 
            Dense(self.hidden_dim, activation='relu'),  
            Dropout(self.dropout_rate), 
            Dense(3)
        ], name="coordinate_predictor")
        
        super(RNAStructurePredictor, self).build(input_shape)
    
    def call(self, inputs, training=False):
        #msa_input, geo_input = inputs
        geo_input = inputs

        #bio_features = self.biology_tower(msa_input, training=training)
        geo_features = self.geometry_tower(geo_input, training=training)

       #tower1_output, tower2_output = bio_features, geo_features
        #for cross_block in self.cross_attention_blocks:
         #   tower1_output, tower2_output = cross_block([tower1_output, tower2_output], training=training)
        
        coordinates = self.coordinate_prediction(geo_features, training=training)
        return coordinates
    

#MSA_DIM = X_train_msa.shape[2]
GEO_DIM = X_train_geo.shape[2]

model = RNAStructurePredictor(MAX_SEQ_LENGTH, 14, GEO_DIM)

#dummy_msa = tf.zeros((1, MAX_SEQ_LENGTH, X_train_msa.shape[2]))
dummy_geo = tf.zeros((1, MAX_SEQ_LENGTH, X_train_geo.shape[2]))
output = model(dummy_geo)
model.summary()


# class TMScore(Metric):
#     def __init__(self, name='tm_score'):
#         super(TMScore, self).__init__(name=name)
#         self.tm_scores = self.add_weight(name='tm_scores', initializer='zeros')
#         self.tm_count = self.add_weight(name='tm_count', initializer='zeros')

#     def result(self):
#         return tf.math.divide_no_nan(self.tm_scores, self.tm_count)
    
#     def reset_state(self):
#         self.tm_scores.assign(0.0)
#         self.tm_count.assign(0.0)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         batch_tm_scores = self.calculate_tm_scores(y_true, y_pred)
#         batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
#         self.tm_scores.assign_add(tf.reduce_sum(batch_tm_scores))
#         self.tm_count.assign_add(batch_size)

#     def calculate_tm_scores(self, y_true, y_pred):
#         batch_size = tf.shape(y_true)[0]
        
#         def calculate_single_tm_score(structures):
#             ref_structure, pred_structure = structures
            
#             L_ref = tf.cast(tf.shape(ref_structure)[0], tf.float32)
#             d0 = self.calculate_d0(L_ref)
#             squared_distances = tf.reduce_sum(tf.square(ref_structure - pred_structure), axis=1)
#             tm_sum = tf.reduce_sum(1.0 / (1.0 + squared_distances / tf.square(d0)))
#             tm_score = (1.0 / L_ref) * tm_sum
#             return tm_score
    
#         return tf.map_fn(calculate_single_tm_score, (y_true, y_pred),fn_output_signature=tf.float32)
    
#     def calculate_d0(self, L_ref):
#         d0_large = 0.6 * tf.sqrt(L_ref - 0.5) - 2.5
        
#         def d0_lt_12(): return tf.constant(0.3, dtype=tf.float32)
#         def d0_12_15(): return tf.constant(0.4, dtype=tf.float32)
#         def d0_16_19(): return tf.constant(0.5, dtype=tf.float32)
#         def d0_20_23(): return tf.constant(0.6, dtype=tf.float32)
#         def d0_24_29(): return tf.constant(0.7, dtype=tf.float32)
        
#         conditions = [(L_ref < 12, d0_lt_12), (tf.logical_and(L_ref >= 12, L_ref <= 15), d0_12_15), (tf.logical_and(L_ref >= 16, L_ref <= 19), d0_16_19), (tf.logical_and(L_ref >= 20, L_ref <= 23), d0_20_23), (tf.logical_and(L_ref >= 24, L_ref <= 29), d0_24_29)]
#         d0_small = tf.case(conditions, default=lambda: tf.constant(0.0, dtype=tf.float32))
#         d0 = tf.cond(L_ref >= 30, lambda: d0_large, lambda: d0_small)
#         return d0

        

# @tf.function
# def RMSD(y_true, y_pred):
#     return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
loss_func = 'mse'
tm_metric = 'mae' #TMScore(name='tm_score')
model.compile(optimizer=optimizer, loss=loss_func, metrics=[tm_metric])


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
tensorboard = TensorBoard(log_dir='logs')
terminate_nan = TerminateOnNaN()
callbacks = [early_stop, reduce_lr, tensorboard, terminate_nan]


if train_sequences is not None:
    history = model.fit(X_train_geo, Y_train_coords, epochs=EPOCHS, validation_data=(X_val_geo, Y_val_coords), callbacks=callbacks, batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE)


def denormalize_coordinates(normalized_coords, normalization_params):
    if normalized_coords.shape[0] == 0:
        return normalized_coords
    
    if normalized_coords.ndim == 1:
        normalized_coords = np.expand_dims(normalized_coords, axis=0)
        
    denormalized_coords = (normalized_coords * normalization_params['std']) + normalization_params['mean']
    
    return denormalized_coords


def create_dummy_submission_file(submission_path='submission.csv', num_predictions=5, num_sequences=10):
    import random
    import string
    
    columns = ['ID', 'resname', 'resid']
    for k in range(1, num_predictions + 1):
        columns.extend([f'x_{k}', f'y_{k}', f'z_{k}'])
    
    rows = []
    nucleotides = ['A', 'U', 'G', 'C']
    
    for seq_idx in range(num_sequences):
        target_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        sequence_length = random.randint(10, 100)
        sequence = ''.join(random.choices(nucleotides, k=sequence_length))
        
        for j, nucleotide in enumerate(sequence):
            row_data = {'ID': f'{target_id}_{j+1}', 'resname': nucleotide, 'resid': j+1}
        
            for k in range(num_predictions):
                row_data[f'x_{k+1}'] = float(np.random.uniform(-50, 50))
                row_data[f'y_{k+1}'] = float(np.random.uniform(-50, 50))
                row_data[f'z_{k+1}'] = float(np.random.uniform(-50, 50))
            
            rows.append(row_data)
    
    submission_df = pd.DataFrame(rows, columns=columns)
    submission_df.to_csv(submission_path, index=False)
    print(f"Dummy submission file saved to {submission_path}")
    print(f"Generated {len(submission_df)} rows from {num_sequences} dummy sequences")
    return submission_df

def create_submission_file(model, test_sequences, submission_path='submission.csv', num_predictions=5):
    X_test_geo, _, _, _ = Geodata(test_sequences, None).prepare_geo_data()
    X_test_msa, _ = MSAData(test_sequences, None, MSA_FOLDER).prepare_msa_data()
    
    columns = ['ID', 'resname', 'resid']
    for k in range(1, num_predictions + 1):
        columns.extend([f'x_{k}', f'y_{k}', f'z_{k}'])
    
    rows = []
    
    for i, (_, row) in enumerate(test_sequences.iterrows()):
        sequence = row['sequence']
        sequence_length = len(sequence)
        target_id = row['target_id']
        
        if i >= len(X_test_geo) or i >= len(X_test_msa):
            print(f"Warning: Index {i} out of bounds for test data")
            continue
            
        sequence_geo_data = np.expand_dims(X_test_geo[i], axis=0)
        sequence_msa_data = np.expand_dims(X_test_msa[i], axis=0)
        
        try:
            #predictions = model.predict([sequence_msa_data, sequence_geo_data], verbose=0)
            predictions = model.predict(sequence_geo_data, verbose=0)
            
            if predictions.ndim == 3:
                predictions = np.squeeze(predictions, axis=0) 
            
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            
            flat_coords = predictions[:sequence_length].flatten()
            lower_bound = np.min(flat_coords)
            higher_bound = np.max(flat_coords)
            
            for j, nucleotide in enumerate(sequence):
                if j >= MAX_SEQ_LENGTH:
                    coords = np.random.uniform(lower_bound, higher_bound, size=(3,))
                else:   
                    coords = predictions[j]
                
                row_data = {'ID': f'{target_id}_{j+1}', 'resname': nucleotide, 'resid': j+1}
                
                try:
                    denormalized_coords = denormalize_coordinates(coords, normalization_params)
                    denormalized_coords = np.squeeze(denormalized_coords)
                    
                    if denormalized_coords.shape == ():
                        denormalized_coords = np.array([denormalized_coords, 0.0, 0.0])
                    elif len(denormalized_coords) < 3:
                        padding = np.zeros(3 - len(denormalized_coords))
                        denormalized_coords = np.concatenate([denormalized_coords, padding])
                    
                    for k in range(num_predictions):
                        noise = np.random.normal(scale=0.5, size=3)
                        row_data[f'x_{k+1}'] = float(denormalized_coords[0] + noise[0])
                        row_data[f'y_{k+1}'] = float(denormalized_coords[1] + noise[1])
                        row_data[f'z_{k+1}'] = float(denormalized_coords[2] + noise[2])
                        
                except Exception as e:
                    print(f"Error processing coordinates for {target_id}_{j+1}: {e}")
                    for k in range(num_predictions):
                        row_data[f'x_{k+1}'] = float(np.random.uniform(-10, 10))
                        row_data[f'y_{k+1}'] = float(np.random.uniform(-10, 10))
                        row_data[f'z_{k+1}'] = float(np.random.uniform(-10, 10))
                
                rows.append(row_data)
                
        except Exception as e:
            print(f"Error predicting for sequence {i} (target_id: {target_id}): {e}")
            for j, nucleotide in enumerate(sequence):
                row_data = {'ID': f'{target_id}_{j+1}', 'resname': nucleotide, 'resid': j+1}
                for k in range(num_predictions):
                    row_data[f'x_{k+1}'] = float(np.random.uniform(-10, 10))
                    row_data[f'y_{k+1}'] = float(np.random.uniform(-10, 10))
                    row_data[f'z_{k+1}'] = float(np.random.uniform(-10, 10))
                rows.append(row_data)
    
    submission_df = pd.DataFrame(rows, columns=columns)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")
    print(f"Generated {len(submission_df)} rows")
    return submission_df


try:
    if test_sequences is not None:
        submission_df = create_submission_file(model, test_sequences, num_predictions=5)
        print("Submission preview:")
        print(submission_df.head(10))
        print(f"Total rows: {len(submission_df)}")
        print(f"submisison shape: {list(submission_df.shape)}")
    else:
        create_dummy_submission_file()
except Exception as e:
    print(f"Error creating submission file: {e}")

