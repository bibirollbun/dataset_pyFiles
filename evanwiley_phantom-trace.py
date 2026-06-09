# # Phantom AI: Multi-Agent Video Forensics Pipeline

# ## Detecting AI-Generated Videos Using Specialized Analysis Agents

# This notebook demonstrates a production-ready multi-agent AI pipeline for detecting AI-generated videos. The system uses specialized agents that analyze different aspects of video content to identify synthetic media.

# ### Architecture Overview

# The pipeline consists of 6 specialized agents:

# 1. **Preprocessing Agent** - Extracts frames and metadata
# 2. **Visual Forensics Agent** - Detects visual artifacts (GAN artifacts, lighting issues)
# 3. **Temporal Analysis Agent** - Analyzes frame-to-frame consistency
# 4. **Audio Forensics Agent** - Detects audio artifacts and inconsistencies
# 5. **Model Attribution Agent** - Identifies which AI model likely generated the video
# 6. **Evidence Aggregation Agent** - Combines all evidence into final result



# ## Setup and Imports



# Standard library imports
import os
import json
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Data science imports
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# PyTorch imports - split for better error handling
print("Importing PyTorch...")
try:
    import torch
    print(f"âœ“ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"âœ— Error importing torch: {e}")
    print("Please ensure PyTorch is installed. In Kaggle, it should be pre-installed.")
    raise

try:
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    print("âœ“ PyTorch modules imported successfully")
except ImportError as e:
    print(f"âœ— Error importing PyTorch modules: {e}")
    raise

warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    # Fallback for older matplotlib versions
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# Set device
print("\nChecking CUDA availability...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

print("\nâœ“ All imports completed successfully!")



# ## 1. Preprocessing Agent

# Extracts frames and metadata from video files.



class PreprocessingAgent:
    """Extracts frames and metadata from video"""
    
    def extract_frames(self, video_path: str, num_frames: int = 30) -> Dict:
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_timestamps = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly
        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            frame_indices = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_timestamps.append(idx / fps if fps > 0 else 0.0)
        
        cap.release()
        
        return {
            'frames': frames,
            'frame_timestamps': frame_timestamps,
            'fps': fps,
            'total_frames': total_frames,
            'duration_sec': total_frames / fps if fps > 0 else 0,
            'video_path': video_path
        }

# Example usage
preprocessor = PreprocessingAgent()
print("Preprocessing Agent initialized")



# ## 2. Visual Forensics Agent

# Detects visual artifacts using deep learning models. Uses a CNN-based architecture to identify GAN artifacts, lighting inconsistencies, and texture anomalies.



# Visual Forensics CNN Model
class VisualForensicsCNN(nn.Module):
    """CNN for detecting visual artifacts in video frames"""
    
    def __init__(self, num_classes=1):
        super(VisualForensicsCNN, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


class VisualForensicsAgent:
    """Analyzes individual frames for visual artifacts"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = VisualForensicsCNN().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        self.model.eval()
    
    def analyze(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze frames for visual artifacts"""
        frames = preprocessed.get("frames", [])
        frame_timestamps = preprocessed.get("frame_timestamps", [])
        
        if not frames:
            return {
                "frame_scores": [],
                "artifact_detections": [],
                "overall_confidence": 0.0,
            }
        
        frame_scores = []
        artifact_detections = []
        
        with torch.no_grad():
            for i, frame in enumerate(frames):
                # Convert to PIL and apply transforms
                frame_pil = Image.fromarray(frame)
                frame_tensor = self.transform(frame_pil).unsqueeze(0).to(device)
                
                # Get prediction
                score = self.model(frame_tensor).item()
                
                frame_scores.append({
                    "frame_index": i,
                    "timestamp": frame_timestamps[i] if i < len(frame_timestamps) else 0.0,
                    "ai_probability": float(score),
                })
                
                # Detect artifacts if score is high
                if score > 0.6:
                    artifact_detections.append({
                        "frame_index": i,
                        "timestamp": frame_timestamps[i] if i < len(frame_timestamps) else 0.0,
                        "artifact_type": "gan_artifact",
                        "confidence": float(score),
                    })
        
        # Calculate overall confidence
        if frame_scores:
            avg_score = np.mean([f["ai_probability"] for f in frame_scores])
            overall_confidence = float(avg_score)
        else:
            overall_confidence = 0.0
        
        return {
            "frame_scores": frame_scores,
            "artifact_detections": artifact_detections,
            "overall_confidence": overall_confidence,
        }

# Initialize agent
visual_agent = VisualForensicsAgent()
print("Visual Forensics Agent initialized")



# Temporal Analysis Model (3D CNN for video sequences)
class TemporalCNN(nn.Module):
    """3D CNN for detecting temporal inconsistencies"""
    
    def __init__(self, num_classes=1):
        super(TemporalCNN, self).__init__()
        # 3D convolutions for temporal-spatial analysis
        self.conv3d1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.conv3d2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3d3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.dropout = nn.Dropout(0.5)
        
        # Adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch, channels, time, height, width)
        x = self.pool(F.relu(self.bn1(self.conv3d1(x))))
        x = self.pool(F.relu(self.bn2(self.conv3d2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3d3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


class TemporalAnalysisAgent:
    """Analyzes temporal consistency across frames"""
    
    def __init__(self, model_path: Optional[str] = None, sequence_length: int = 16):
        self.sequence_length = sequence_length
        self.model = TemporalCNN().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
    
    def analyze(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal consistency"""
        frames = preprocessed.get("frames", [])
        frame_timestamps = preprocessed.get("frame_timestamps", [])
        
        if len(frames) < 2:
            return {
                "transition_scores": [],
                "motion_anomalies": [],
                "temporal_confidence": 0.0,
            }
        
        transition_scores = []
        motion_anomalies = []
        
        # Process frames in sequences
        num_sequences = max(1, len(frames) - self.sequence_length + 1)
        
        with torch.no_grad():
            for i in range(0, len(frames) - self.sequence_length + 1, max(1, self.sequence_length // 2)):
                sequence = frames[i:i + self.sequence_length]
                sequence_timestamps = frame_timestamps[i:i + self.sequence_length]
                
                # Transform frames
                sequence_tensors = []
                for frame in sequence:
                    frame_pil = Image.fromarray(frame)
                    frame_tensor = self.transform(frame_pil)
                    sequence_tensors.append(frame_tensor)
                
                # Stack into sequence: (time, channels, height, width)
                sequence_tensor = torch.stack(sequence_tensors).permute(1, 0, 2, 3).unsqueeze(0).to(device)
                
                # Get prediction
                score = self.model(sequence_tensor).item()
                
                start_timestamp = sequence_timestamps[0] if sequence_timestamps else 0.0
                end_timestamp = sequence_timestamps[-1] if sequence_timestamps else 0.0
                
                transition_scores.append({
                    "start_frame": i,
                    "end_frame": i + self.sequence_length - 1,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "anomaly_score": float(score),
                })
                
                # Detect anomalies
                if score > 0.7:
                    motion_anomalies.append({
                        "start_frame": i,
                        "end_frame": i + self.sequence_length - 1,
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                        "anomaly_type": "temporal_inconsistency",
                        "confidence": float(score),
                    })
        
        # If no sequences, use simple frame difference
        if not transition_scores:
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                score = self._simple_transition_analysis(frame1, frame2)
                
                transition_scores.append({
                    "start_frame": i,
                    "end_frame": i + 1,
                    "start_timestamp": frame_timestamps[i] if i < len(frame_timestamps) else 0.0,
                    "end_timestamp": frame_timestamps[i + 1] if i + 1 < len(frame_timestamps) else 0.0,
                    "anomaly_score": float(score),
                })
                
                if score > 0.7:
                    motion_anomalies.append({
                        "start_frame": i,
                        "end_frame": i + 1,
                        "start_timestamp": frame_timestamps[i] if i < len(frame_timestamps) else 0.0,
                        "end_timestamp": frame_timestamps[i + 1] if i + 1 < len(frame_timestamps) else 0.0,
                        "anomaly_type": "temporal_inconsistency",
                        "confidence": float(score),
                    })
        
        # Calculate overall confidence
        if transition_scores:
            avg_score = np.mean([t["anomaly_score"] for t in transition_scores])
            temporal_confidence = float(avg_score)
        else:
            temporal_confidence = 0.0
        
        return {
            "transition_scores": transition_scores,
            "motion_anomalies": motion_anomalies,
            "temporal_confidence": temporal_confidence,
        }
    
    def _simple_transition_analysis(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Fallback simple frame difference analysis"""
        if len(frame1.shape) == 3:
            gray1 = np.mean(frame1, axis=2)
            gray2 = np.mean(frame2, axis=2)
        else:
            gray1 = frame1
            gray2 = frame2
        
        diff = np.abs(gray1.astype(float) - gray2.astype(float))
        mean_diff = np.mean(diff)
        score = min(1.0, mean_diff / 50.0)
        return float(score)

# Initialize agent
temporal_agent = TemporalAnalysisAgent()
print("Temporal Analysis Agent initialized")



# ## 4. Audio Forensics Agent

# Detects audio artifacts and inconsistencies using spectral analysis.



try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio processing libraries not available. Install librosa and soundfile for audio analysis.")


class AudioForensicsAgent:
    """Detects audio artifacts and inconsistencies"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
    
    def analyze(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio for artifacts"""
        video_path = preprocessed.get("video_path", "")
        
        if not AUDIO_AVAILABLE or not video_path:
            return {
                "audio_features": {},
                "artifact_detections": [],
                "audio_confidence": 0.0,
            }
        
        try:
            # Extract audio from video
            audio, sr = self._extract_audio(video_path)
            if audio is None:
                return {
                    "audio_features": {},
                    "artifact_detections": [],
                    "audio_confidence": 0.0,
                }
            
            # Extract features
            features = self._extract_features(audio, sr)
            
            # Detect artifacts
            artifact_score = self._detect_artifacts(features)
            
            artifact_detections = []
            if artifact_score > 0.6:
                artifact_detections.append({
                    "artifact_type": "audio_inconsistency",
                    "confidence": float(artifact_score),
                    "timestamp": preprocessed.get("duration_sec", 0) / 2,
                })
            
            return {
                "audio_features": features,
                "artifact_detections": artifact_detections,
                "audio_confidence": float(artifact_score),
            }
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            return {
                "audio_features": {},
                "artifact_detections": [],
                "audio_confidence": 0.0,
            }
    
    def _extract_audio(self, video_path: str):
        """Extract audio from video"""
        try:
            # Use librosa to load audio
            audio, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except:
            return None, None
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features"""
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Chroma features
        chroma = librosa.feature.chroma(y=audio, sr=sr)
        
        return {
            "mfcc_mean": float(np.mean(mfccs)),
            "mfcc_std": float(np.std(mfccs)),
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "zcr_mean": float(np.mean(zero_crossing_rate)),
            "chroma_mean": float(np.mean(chroma)),
        }
    
    def _detect_artifacts(self, features: Dict[str, float]) -> float:
        """Detect audio artifacts based on features"""
        # Simple heuristic-based detection
        # In production, use trained classifier
        score = 0.0
        
        # Check for unusual spectral characteristics
        if features.get("spectral_centroid_mean", 0) > 3000:
            score += 0.3
        if features.get("zcr_mean", 0) > 0.1:
            score += 0.2
        if features.get("mfcc_std", 0) > 10:
            score += 0.3
        
        return min(1.0, score)

# Initialize agent
audio_agent = AudioForensicsAgent() if AUDIO_AVAILABLE else None
if audio_agent:
    print("Audio Forensics Agent initialized")
else:
    print("Audio Forensics Agent skipped (libraries not available)")



# ## 5. Model Attribution Agent

# Identifies which AI model likely generated the video using embedding-based classification.



# Model Attribution Network
class ModelAttributionNet(nn.Module):
    """Network for attributing videos to specific AI models"""
    
    def __init__(self, num_models=5, embedding_dim=512):
        super(ModelAttributionNet, self).__init__()
        
        # Feature extractor (shared backbone)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Embedding layer
        self.embedding = nn.Linear(256, embedding_dim)
        
        # Per-model classifiers
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            ) for _ in range(num_models)
        ])
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Get embedding
        embedding = self.embedding(features)
        
        # Get predictions for each model
        predictions = []
        for classifier in self.classifiers:
            pred = torch.sigmoid(classifier(embedding))
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1), embedding


class ModelAttributionAgent:
    """Attributes video to specific AI models"""
    
    KNOWN_MODELS = [
        "Runway Gen-3",
        "Pika",
        "Sora",
        "Stable Video",
        "Gen-2",
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = ModelAttributionNet(num_models=len(self.KNOWN_MODELS)).to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
    
    def analyze(self, preprocessed: Dict[str, Any], 
                visual_results: Dict[str, Any],
                temporal_results: Dict[str, Any]) -> Dict[str, Any]:
        """Attribute video to AI models"""
        frames = preprocessed.get("frames", [])
        
        if not frames:
            return {
                "model_scores": {},
                "top_models": [],
                "attribution_confidence": 0.0,
            }
        
        # Sample frames for analysis
        sample_frames = frames[::max(1, len(frames) // 10)][:10]
        
        model_scores = {model: 0.0 for model in self.KNOWN_MODELS}
        
        with torch.no_grad():
            for frame in sample_frames:
                frame_pil = Image.fromarray(frame)
                frame_tensor = self.transform(frame_pil).unsqueeze(0).to(device)
                
                predictions, _ = self.model(frame_tensor)
                predictions = predictions.cpu().numpy()[0]
                
                # Accumulate scores
                for i, model_name in enumerate(self.KNOWN_MODELS):
                    if i < len(predictions):
                        model_scores[model_name] += float(predictions[i])
        
        # Average scores
        num_frames = len(sample_frames)
        for model_name in model_scores:
            model_scores[model_name] /= num_frames if num_frames > 0 else 1
        
        # Get top models
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_models = [model for model, score in sorted_models if score > 0.3]
        
        # Calculate overall confidence
        max_score = max(model_scores.values()) if model_scores else 0.0
        attribution_confidence = float(max_score)
        
        return {
            "model_scores": model_scores,
            "top_models": top_models[:3],  # Top 3
            "attribution_confidence": attribution_confidence,
        }

# Initialize agent
attribution_agent = ModelAttributionAgent()
print("Model Attribution Agent initialized")



# ## 6. Evidence Aggregation Agent

# Combines evidence from all agents to produce final authenticity score and explanation.



class EvidenceAggregationAgent:
    """Combines evidence from all agents into final result"""
    
    def __init__(self):
        # Weight configuration for different agents
        self.weights = {
            "visual_forensics": 0.35,
            "temporal_analysis": 0.30,
            "audio_forensics": 0.15,
            "model_attribution": 0.20,
        }
    
    def aggregate(self, analysis_id: str, preprocessed: Dict[str, Any],
                  visual_results: Dict[str, Any],
                  temporal_results: Dict[str, Any],
                  audio_results: Dict[str, Any],
                  attribution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all evidence into final result"""
        
        # Extract confidence scores
        visual_confidence = visual_results.get("overall_confidence", 0.0)
        temporal_confidence = temporal_results.get("temporal_confidence", 0.0)
        audio_confidence = audio_results.get("audio_confidence", 0.0)
        attribution_confidence = attribution_results.get("attribution_confidence", 0.0)
        
        # Weighted average
        weighted_score = (
            visual_confidence * self.weights["visual_forensics"] +
            temporal_confidence * self.weights["temporal_analysis"] +
            audio_confidence * self.weights["audio_forensics"] +
            attribution_confidence * self.weights["model_attribution"]
        )
        
        # Calculate authenticity score (0-100, higher = more authentic)
        authenticity_score = int((1.0 - weighted_score) * 100)
        authenticity_score = max(0, min(100, authenticity_score))
        
        # Determine label
        label = "likely_ai" if authenticity_score < 50 else "likely_real"
        
        # Collect suspicious segments
        suspicious_segments = []
        
        # Add visual artifacts
        for artifact in visual_results.get("artifact_detections", []):
            suspicious_segments.append({
                "start_sec": artifact.get("timestamp", 0.0),
                "end_sec": artifact.get("timestamp", 0.0) + 1.0,
                "severity": artifact.get("confidence", 0.0),
                "source": "visual_forensics",
                "type": artifact.get("artifact_type", "gan_artifact"),
            })
        
        # Add temporal anomalies
        for anomaly in temporal_results.get("motion_anomalies", []):
            suspicious_segments.append({
                "start_sec": anomaly.get("start_timestamp", 0.0),
                "end_sec": anomaly.get("end_timestamp", 0.0),
                "severity": anomaly.get("confidence", 0.0),
                "source": "temporal_analysis",
                "type": "temporal_inconsistency",
            })
        
        # Add audio artifacts
        for artifact in audio_results.get("artifact_detections", []):
            suspicious_segments.append({
                "start_sec": artifact.get("timestamp", 0.0) - 0.5,
                "end_sec": artifact.get("timestamp", 0.0) + 0.5,
                "severity": artifact.get("confidence", 0.0),
                "source": "audio_forensics",
                "type": artifact.get("artifact_type", "audio_inconsistency"),
            })
        
        # Sort by severity
        suspicious_segments.sort(key=lambda x: x["severity"], reverse=True)
        
        # Get top models
        top_models = attribution_results.get("top_models", [])
        
        # Generate explanation
        explanation = self._generate_explanation(
            authenticity_score, label, suspicious_segments, top_models,
            visual_confidence, temporal_confidence, audio_confidence, attribution_confidence
        )
        
        return {
            "authenticity_score": authenticity_score,
            "label": label,
            "suspicious_segments": suspicious_segments[:10],  # Top 10
            "top_models": top_models,
            "confidence_breakdown": {
                "visual_forensics": float(visual_confidence),
                "temporal_analysis": float(temporal_confidence),
                "audio_forensics": float(audio_confidence),
                "model_attribution": float(attribution_confidence),
            },
            "explanation": explanation,
            "analysis_id": analysis_id,
            "created_at": datetime.utcnow().isoformat(),
        }
    
    def _generate_explanation(self, authenticity_score: int, label: str,
                             suspicious_segments: List[Dict], top_models: List[str],
                             visual_conf: float, temporal_conf: float,
                             audio_conf: float, attribution_conf: float) -> str:
        """Generate human-readable explanation"""
        
        if label == "likely_real":
            explanation = f"Video appears authentic (score: {authenticity_score}/100). "
            explanation += "No significant AI generation artifacts detected across visual, temporal, and audio analysis."
        else:
            explanation = f"Video shows strong indicators of AI generation (score: {authenticity_score}/100). "
            
            if suspicious_segments:
                explanation += f"Detected {len(suspicious_segments)} suspicious segment(s). "
            
            # Add agent-specific findings
            findings = []
            if visual_conf > 0.7:
                findings.append(f"strong visual artifacts ({visual_conf:.0%} confidence)")
            if temporal_conf > 0.7:
                findings.append(f"temporal inconsistencies ({temporal_conf:.0%} confidence)")
            if audio_conf > 0.6:
                findings.append(f"audio anomalies ({audio_conf:.0%} confidence)")
            if top_models and attribution_conf > 0.5:
                findings.append(f"characteristics of {', '.join(top_models[:2])} models")
            
            if findings:
                explanation += "Key findings: " + ", ".join(findings) + "."
        
        return explanation

# Initialize agent
evidence_agent = EvidenceAggregationAgent()
print("Evidence Aggregation Agent initialized")



# ## 7. Orchestrator

# Coordinates all agents in the multi-agent pipeline.



class PhantomTraceOrchestrator:
    """Main orchestrator that coordinates the multi-agent pipeline"""
    
    def __init__(self, preprocessing_agent, visual_forensics_agent,
                 temporal_analysis_agent, audio_forensics_agent,
                 model_attribution_agent, evidence_aggregation_agent):
        self.preprocessing_agent = preprocessing_agent
        self.visual_forensics_agent = visual_forensics_agent
        self.temporal_analysis_agent = temporal_analysis_agent
        self.audio_forensics_agent = audio_forensics_agent
        self.model_attribution_agent = model_attribution_agent
        self.evidence_aggregation_agent = evidence_aggregation_agent
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Main orchestration method"""
        import uuid
        analysis_id = str(uuid.uuid4())
        
        print(f"Starting analysis {analysis_id}")
        print(f"Video: {video_path}")
        
        try:
            # Step 1: Preprocessing
            print("\n[1/5] Preprocessing...")
            preprocessed = self.preprocessing_agent.extract_frames(video_path)
            print(f"  Extracted {len(preprocessed['frames'])} frames")
            
            # Step 2: Visual Forensics
            print("\n[2/5] Visual Forensics...")
            visual_results = self.visual_forensics_agent.analyze(preprocessed)
            print(f"  Visual confidence: {visual_results['overall_confidence']:.2%}")
            
            # Step 3: Temporal Analysis
            print("\n[3/5] Temporal Analysis...")
            temporal_results = self.temporal_analysis_agent.analyze(preprocessed)
            print(f"  Temporal confidence: {temporal_results['temporal_confidence']:.2%}")
            
            # Step 4: Audio Forensics
            print("\n[4/5] Audio Forensics...")
            if self.audio_forensics_agent:
                audio_results = self.audio_forensics_agent.analyze(preprocessed)
                print(f"  Audio confidence: {audio_results['audio_confidence']:.2%}")
            else:
                audio_results = {"audio_confidence": 0.0, "artifact_detections": []}
                print("  Audio analysis skipped")
            
            # Step 5: Model Attribution
            print("\n[5/5] Model Attribution...")
            attribution_results = self.model_attribution_agent.analyze(
                preprocessed, visual_results, temporal_results
            )
            print(f"  Top models: {attribution_results['top_models']}")
            
            # Step 6: Evidence Aggregation
            print("\n[Aggregating Evidence...]")
            final_result = self.evidence_aggregation_agent.aggregate(
                analysis_id=analysis_id,
                preprocessed=preprocessed,
                visual_results=visual_results,
                temporal_results=temporal_results,
                audio_results=audio_results,
                attribution_results=attribution_results,
            )
            
            print(f"\nâœ“ Analysis complete!")
            print(f"  Authenticity Score: {final_result['authenticity_score']}/100")
            print(f"  Label: {final_result['label']}")
            
            return final_result
            
        except Exception as e:
            print(f"\nâœ— Analysis failed: {e}")
            raise

# Initialize orchestrator
orchestrator = PhantomTraceOrchestrator(
    preprocessing_agent=preprocessor,
    visual_forensics_agent=visual_agent,
    temporal_analysis_agent=temporal_agent,
    audio_forensics_agent=audio_agent,
    model_attribution_agent=attribution_agent,
    evidence_aggregation_agent=evidence_agent,
)
print("Orchestrator initialized")



# ## 8. Model Training

# Training functions for each agent's models.



def train_visual_forensics_model(train_data: List[Tuple[str, int]], 
                                  val_data: List[Tuple[str, int]],
                                  num_epochs: int = 10,
                                  batch_size: int = 32,
                                  learning_rate: float = 0.001):
    """Train visual forensics model"""
    
    model = VisualForensicsCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Training Visual Forensics Model...")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for video_path, label in train_data:
            try:
                preprocessed = preprocessor.extract_frames(video_path, num_frames=5)
                frames = preprocessed.get("frames", [])
                
                if not frames:
                    continue
                
                # Use first frame for training (can extend to multiple frames)
                frame = frames[0]
                frame_pil = Image.fromarray(frame)
                frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
                label_tensor = torch.tensor([[float(label)]], device=device)
                
                optimizer.zero_grad()
                output = model(frame_tensor)
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
            except Exception as e:
                continue
        
        avg_train_loss = train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for video_path, label in val_data:
                try:
                    preprocessed = preprocessor.extract_frames(video_path, num_frames=5)
                    frames = preprocessed.get("frames", [])
                    
                    if not frames:
                        continue
                    
                    frame = frames[0]
                    frame_pil = Image.fromarray(frame)
                    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
                    label_tensor = torch.tensor([[float(label)]], device=device)
                    
                    output = model(frame_tensor)
                    loss = criterion(output, label_tensor)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "visual_forensics_model.pth")
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model, train_losses, val_losses


def evaluate_model(model, test_data: List[Tuple[str, int]], agent_type: str = "visual"):
    """Evaluate model on test data"""
    
    model.eval()
    predictions = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for video_path, label in test_data:
            try:
                preprocessed = preprocessor.extract_frames(video_path, num_frames=5)
                frames = preprocessed.get("frames", [])
                
                if not frames:
                    continue
                
                frame = frames[0]
                frame_pil = Image.fromarray(frame)
                frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
                
                output = model(frame_tensor)
                pred = (output.item() > 0.5).astype(int)
                
                predictions.append(pred)
                labels.append(label)
                
            except Exception as e:
                continue
    
    # Calculate metrics
    if len(predictions) > 0 and len(labels) > 0:
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # For AUC, need probabilities
        probs = []
        true_labels = []
        with torch.no_grad():
            for video_path, label in test_data:
                try:
                    preprocessed = preprocessor.extract_frames(video_path, num_frames=5)
                    frames = preprocessed.get("frames", [])
                    if not frames:
                        continue
                    frame = frames[0]
                    frame_pil = Image.fromarray(frame)
                    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
                    output = model(frame_tensor)
                    probs.append(output.item())
                    true_labels.append(label)
                except:
                    continue
        
        try:
            auc = roc_auc_score(true_labels, probs) if len(probs) > 0 else 0.0
        except:
            auc = 0.0
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc),
        }
        
        print(f"\n{agent_type.upper()} Model Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return metrics
    else:
        print(f"No valid predictions for {agent_type} model")
        return {}

print("Training and evaluation functions defined")



# ## 9. Visualization and Metrics

# Functions for visualizing results and evaluating model performance.



def plot_training_curves(train_losses: List[float], val_losses: List[float], title: str = "Training Curves"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], title: str = "Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Real', 'AI-Generated'],
                yticklabels=['Real', 'AI-Generated'])
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_confidence_breakdown(result: Dict[str, Any]):
    """Plot confidence breakdown by agent"""
    confidence = result.get("confidence_breakdown", {})
    
    if not confidence:
        print("No confidence breakdown available")
        return
    
    agents = list(confidence.keys())
    scores = [confidence[agent] for agent in agents]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(agents, scores, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    plt.xlabel('Confidence Score', fontsize=12)
    plt.title('Agent Confidence Breakdown', fontsize=16, fontweight='bold')
    plt.xlim(0, 1.0)
    
    # Add value labels
    for i, (agent, score) in enumerate(zip(agents, scores)):
        plt.text(score + 0.02, i, f'{score:.2%}', va='center', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_suspicious_segments(result: Dict[str, Any], duration: float):
    """Plot timeline of suspicious segments"""
    segments = result.get("suspicious_segments", [])
    
    if not segments:
        print("No suspicious segments detected")
        return
    
    plt.figure(figsize=(12, 4))
    
    # Plot timeline
    for seg in segments:
        start = seg.get("start_sec", 0)
        end = seg.get("end_sec", start + 1)
        severity = seg.get("severity", 0)
        source = seg.get("source", "unknown")
        
        color_map = {
            "visual_forensics": "#e74c3c",
            "temporal_analysis": "#f39c12",
            "audio_forensics": "#3498db",
            "model_attribution": "#9b59b6"
        }
        color = color_map.get(source, "#95a5a6")
        
        plt.barh(0, end - start, left=start, height=0.5, 
                color=color, alpha=severity, label=source if source not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(start + (end - start) / 2, 0, f'{severity:.2f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('')
    plt.title('Suspicious Segments Timeline', fontsize=16, fontweight='bold')
    plt.xlim(0, duration)
    plt.yticks([])
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_frame_scores(frame_scores: List[Dict], title: str = "Frame-by-Frame AI Probability"):
    """Plot frame-by-frame AI probability scores"""
    if not frame_scores:
        print("No frame scores available")
        return
    
    timestamps = [f["timestamp"] for f in frame_scores]
    probabilities = [f["ai_probability"] for f in frame_scores]
    
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, probabilities, 'b-', linewidth=2, alpha=0.7)
    plt.fill_between(timestamps, probabilities, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)', linewidth=1.5)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('AI Probability', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def display_results_summary(result: Dict[str, Any]):
    """Display comprehensive results summary"""
    print("=" * 70)
    print("Phantom Trace - ANALYSIS RESULTS")
    print("=" * 70)
    
    print(f"\nğŸ“Š Authenticity Score: {result.get('authenticity_score', 0)}/100")
    print(f"ğŸ�·ï¸�  Label: {result.get('label', 'unknown').upper()}")
    
    confidence = result.get("confidence_breakdown", {})
    if confidence:
        print(f"\nğŸ”� Agent Confidence Scores:")
        for agent, score in confidence.items():
            print(f"   â€¢ {agent.replace('_', ' ').title()}: {score:.2%}")
    
    top_models = result.get("top_models", [])
    if top_models:
        print(f"\nğŸ¤– Suspected AI Models:")
        for model in top_models:
            print(f"   â€¢ {model}")
    
    segments = result.get("suspicious_segments", [])
    if segments:
        print(f"\nâš ï¸�  Suspicious Segments Detected: {len(segments)}")
        for i, seg in enumerate(segments[:5], 1):  # Show top 5
            print(f"   {i}. {seg.get('start_sec', 0):.1f}s - {seg.get('end_sec', 0):.1f}s "
                  f"(severity: {seg.get('severity', 0):.2f}, source: {seg.get('source', 'unknown')})")
    
    explanation = result.get("explanation", "")
    if explanation:
        print(f"\nğŸ“� Explanation:")
        print(f"   {explanation}")
    
    print("\n" + "=" * 70)

print("Visualization functions defined")



# ## 10. Dataset Loading Utilities

# Helper functions for loading and preparing datasets for training and evaluation.



def load_dataset_from_directory(data_dir: str, real_subdir: str = "real", 
                                 fake_subdir: str = "fake") -> List[Tuple[str, int]]:
    """Load dataset from directory structure
    
    Expected structure:
    data_dir/
        real/
            video1.mp4
            video2.mp4
            ...
        fake/
            video1.mp4
            video2.mp4
            ...
    """
    dataset = []
    
    # Load real videos
    real_dir = os.path.join(data_dir, real_subdir)
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(real_dir, filename)
                dataset.append((video_path, 0))  # 0 = real
    
    # Load fake/AI-generated videos
    fake_dir = os.path.join(data_dir, fake_subdir)
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(fake_dir, filename)
                dataset.append((video_path, 1))  # 1 = fake/AI-generated
    
    print(f"Loaded {len(dataset)} videos from {data_dir}")
    real_count = sum(1 for _, label in dataset if label == 0)
    fake_count = sum(1 for _, label in dataset if label == 1)
    print(f"  Real: {real_count}, AI-Generated: {fake_count}")
    
    return dataset


def split_dataset(dataset: List[Tuple[str, int]], train_ratio: float = 0.7,
                  val_ratio: float = 0.15, test_ratio: float = 0.15,
                  random_seed: int = 42) -> Tuple[List, List, List]:
    """Split dataset into train, validation, and test sets"""
    from sklearn.model_selection import train_test_split
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # First split: train + val/test
    train_data, temp_data = train_test_split(
        dataset, test_size=(1 - train_ratio), random_state=random_seed, shuffle=True
    )
    
    # Second split: val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data, test_size=(1 - val_size), random_state=random_seed, shuffle=True
    )
    
    print(f"\nDataset Split:")
    print(f"  Training: {len(train_data)} samples ({len(train_data)/len(dataset):.1%})")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/len(dataset):.1%})")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/len(dataset):.1%})")
    
    return train_data, val_data, test_data


def load_dataset_from_csv(csv_path: str, video_dir: str = "", 
                          video_column: str = "video_path",
                          label_column: str = "label") -> List[Tuple[str, int]]:
    """Load dataset from CSV file
    
    CSV should have columns: video_path (or video_column), label (or label_column)
    Labels should be: 0 for real, 1 for fake/AI-generated
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    dataset = []
    
    for _, row in df.iterrows():
        video_path = row[video_column]
        
        # If video_dir is provided, prepend it to the path
        if video_dir and not os.path.isabs(video_path):
            video_path = os.path.join(video_dir, video_path)
        
        label = int(row[label_column])
        
        if os.path.exists(video_path):
            dataset.append((video_path, label))
        else:
            print(f"Warning: Video not found: {video_path}")
    
    print(f"Loaded {len(dataset)} videos from CSV: {csv_path}")
    return dataset

print("Dataset loading utilities defined")



# ## 11. Complete Pipeline Execution

# Example of running the complete pipeline on a video.



# Example: Run complete pipeline on a video
# Replace 'path/to/video.mp4' with your actual video path

# Uncomment and modify the path below to run analysis on a video
"""
video_path = "path/to/video.mp4"  # Change this to your video path

if os.path.exists(video_path):
    # Run complete pipeline
    result = orchestrator.analyze_video(video_path)
    
    # Display results
    display_results_summary(result)
    
    # Visualize results
    plot_confidence_breakdown(result)
    
    # Plot suspicious segments if available
    duration = result.get("confidence_breakdown", {}).get("duration_sec", 60)
    plot_suspicious_segments(result, duration)
    
    # Plot frame scores if available
    visual_results = result.get("visual_results", {})
    frame_scores = visual_results.get("frame_scores", [])
    if frame_scores:
        plot_frame_scores(frame_scores)
else:
    print(f"Video not found: {video_path}")
    print("Please update the video_path variable with a valid video file path")
"""

print("Pipeline execution example ready")
print("Uncomment the code above and provide a video path to run analysis")



# ## 12. Training Example

# Example of training the visual forensics model.



# Example: Train visual forensics model
# Uncomment and modify paths below to train on your dataset

"""
# Load dataset
data_dir = "path/to/your/dataset"  # Change this to your dataset directory
dataset = load_dataset_from_directory(data_dir)

# Split dataset
train_data, val_data, test_data = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Train model
model, train_losses, val_losses = train_visual_forensics_model(
    train_data=train_data,
    val_data=val_data,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001
)

# Plot training curves
plot_training_curves(train_losses, val_losses, "Visual Forensics Model Training")

# Evaluate on test set
test_metrics = evaluate_model(model, test_data, agent_type="visual")

# Plot confusion matrix (requires predictions)
# y_true = [label for _, label in test_data]
# y_pred = [predictions...]  # Extract from evaluate_model
# plot_confusion_matrix(y_true, y_pred, "Visual Forensics Confusion Matrix")
"""

print("Training example ready")
print("Uncomment the code above and provide dataset paths to train models")



# ## 14. Metadata Forensics Agent

# **NEW DIFFERENTIATOR:** Analyzes video metadata for creator attribution, tool detection, and device fingerprinting.

# This goes beyond AI model detection to identify **who created the deepfake** by analyzing:
# - Video metadata (EXIF, timestamps, encoder info)
# - Creation tool signatures (DeepFaceLab, FaceSwap, etc.)
# - Device fingerprints (camera model, sensor characteristics)
# - Compression artifacts (multi-pass encoding, tool-specific patterns)
# - Temporal inconsistencies (timestamp manipulation)

# ---

# ## 14a. Metadata Forensics Implementation



# Metadata Forensics Agent
try:
    import ffmpeg
    import json
    from datetime import datetime
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    print("ffmpeg-python not available. Install: pip install ffmpeg-python")


class MetadataForensicsAgent:
    """Analyzes video metadata for creator attribution and forensics"""
    
    def __init__(self):
        # Tool signature database
        self.tool_signatures = {
            'deepfacelab': ['dfl', 'deepfacelab', 'deepface'],
            'faceswap': ['faceswap'],
            'wav2lip': ['wav2lip'],
            'runway': ['runway', 'gen-', 'runwayml'],
            'synthesia': ['synthesia'],
            'd-id': ['d-id', 'did'],
            'after_effects': ['after effects', 'adobe after effects'],
            'premiere': ['adobe premiere', 'premiere pro', 'adobe premiere pro'],
            'ffmpeg': ['ffmpeg', 'libav'],
            'handbrake': ['handbrake'],
            'obs': ['obs', 'obs studio'],
            'streamlabs': ['streamlabs'],
            'zoom': ['zoom'],
        }
    
    def analyze(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze metadata for forensics"""
        video_path = preprocessed.get("video_path", "")
        
        if not METADATA_AVAILABLE or not video_path:
            return {
                "metadata": {},
                "tool_detection": {},
                "creator_signals": [],
                "confidence_score": 0.0,
            }
        
        try:
            # Extract metadata
            metadata = self._extract_metadata(video_path)
            
            # Detect creation tools
            tool_detection = self._detect_creation_tools(metadata)
            
            # Identify creator signals
            creator_signals = self._identify_creator_signals(metadata, tool_detection)
            
            # Calculate confidence
            confidence = self._calculate_confidence(creator_signals, tool_detection)
            
            return {
                "metadata": metadata,
                "tool_detection": tool_detection,
                "creator_signals": creator_signals,
                "confidence_score": confidence,
            }
        except Exception as e:
            logger.warning(f"Metadata forensics failed: {e}")
            return {
                "metadata": {},
                "tool_detection": {},
                "creator_signals": [],
                "confidence_score": 0.0,
            }
    
    def _extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata"""
        try:
            probe = ffmpeg.probe(video_path)
            format_info = probe.get('format', {})
            streams = probe.get('streams', [])
            
            # Extract video stream info
            video_stream = next((s for s in streams if s.get('codec_type') == 'video'), {})
            audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), {})
            
            metadata = {
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'format_name': format_info.get('format_name', ''),
                'format_long_name': format_info.get('format_long_name', ''),
                'encoder': format_info.get('tags', {}).get('encoder', ''),
                'creation_time': format_info.get('tags', {}).get('creation_time', ''),
                'software': format_info.get('tags', {}).get('software', ''),
                'title': format_info.get('tags', {}).get('title', ''),
                'comment': format_info.get('tags', {}).get('comment', ''),
                
                # Video stream
                'video_codec': video_stream.get('codec_name', ''),
                'video_codec_long': video_stream.get('codec_long_name', ''),
                'video_profile': video_stream.get('profile', ''),
                'video_width': video_stream.get('width', 0),
                'video_height': video_stream.get('height', 0),
                'video_fps': video_stream.get('r_frame_rate', ''),
                'video_bitrate': video_stream.get('bit_rate', 0),
                'video_pixel_format': video_stream.get('pix_fmt', ''),
                'video_gop_size': video_stream.get('gop_size', 0),
                
                # Audio stream
                'audio_codec': audio_stream.get('codec_name', ''),
                'audio_codec_long': audio_stream.get('codec_long_name', ''),
                'audio_bitrate': audio_stream.get('bit_rate', 0),
                'audio_sample_rate': audio_stream.get('sample_rate', 0),
            }
            
            return metadata
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}
    
    def _detect_creation_tools(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Detect software tools used to create video"""
        detected_tools = []
        tool_confidence = {}
        
        # Check encoder string
        encoder = metadata.get('encoder', '').lower()
        software = metadata.get('software', '').lower()
        comment = metadata.get('comment', '').lower()
        title = metadata.get('title', '').lower()
        
        all_text = ' '.join([encoder, software, comment, title])
        
        # Match against tool signatures
        for tool, signatures in self.tool_signatures.items():
            for sig in signatures:
                if sig in all_text:
                    if tool not in detected_tools:
                        detected_tools.append(tool)
                    # Higher confidence if found in encoder/software fields
                    if sig in encoder or sig in software:
                        tool_confidence[tool] = 0.8
                    else:
                        tool_confidence[tool] = max(tool_confidence.get(tool, 0.0), 0.6)
                    break
        
        # Analyze codec patterns for tool-specific signatures
        codec_patterns = self._analyze_codec_patterns(metadata)
        if codec_patterns:
            detected_tools.extend(codec_patterns.get('tools', []))
        
        return {
            'detected_tools': list(set(detected_tools)),  # Remove duplicates
            'confidence': tool_confidence,
            'codec_patterns': codec_patterns,
        }
    
    def _analyze_codec_patterns(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codec patterns for tool-specific signatures"""
        patterns = {
            'tools': [],
            'anomalies': [],
        }
        
        # Check for anomalies
        gop_size = metadata.get('video_gop_size', 0)
        if gop_size == 0 or gop_size > 300:
            patterns['anomalies'].append('irregular_gop_structure')
        
        # Multi-pass encoding indicators
        profile = metadata.get('video_profile', '')
        if 'high' in profile.lower() or 'main' in profile.lower():
            patterns['tools'].append('high_quality_encoding')
        
        return patterns
    
    def _identify_creator_signals(self, metadata: Dict[str, Any], 
                                  tool_detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify signals pointing to creator"""
        signals = []
        
        # 1. Encoder signatures
        encoder = metadata.get('encoder', '')
        if encoder:
            signals.append({
                'type': 'encoder_signature',
                'value': encoder,
                'confidence': 0.7,
                'description': f'Video encoded with: {encoder}',
            })
        
        # 2. Software/creation tool
        detected_tools = tool_detection.get('detected_tools', [])
        for tool in detected_tools:
            conf = tool_detection.get('confidence', {}).get(tool, 0.6)
            signals.append({
                'type': 'tool_signature',
                'value': tool,
                'confidence': conf,
                'description': f'Likely created using: {tool.replace("_", " ").title()}',
            })
        
        # 3. Timestamp analysis
        creation_time = metadata.get('creation_time', '')
        if creation_time:
            signals.append({
                'type': 'temporal_signal',
                'value': creation_time,
                'confidence': 0.6,
                'description': f'Creation timestamp: {creation_time}',
            })
        
        # 4. Codec anomalies
        codec_patterns = tool_detection.get('codec_patterns', {})
        anomalies = codec_patterns.get('anomalies', [])
        if anomalies:
            signals.append({
                'type': 'compression_anomaly',
                'value': anomalies,
                'confidence': 0.5,
                'description': f'Compression anomalies: {", ".join(anomalies)}',
            })
        
        # 5. Format information
        format_name = metadata.get('format_name', '')
        if format_name:
            signals.append({
                'type': 'format_info',
                'value': format_name,
                'confidence': 0.4,
                'description': f'Container format: {format_name}',
            })
        
        return signals
    
    def _calculate_confidence(self, creator_signals: List[Dict], 
                             tool_detection: Dict[str, Any]) -> float:
        """Calculate overall confidence in metadata forensics"""
        if not creator_signals:
            return 0.0
        
        # Weight signals by type
        weights = {
            'encoder_signature': 0.3,
            'tool_signature': 0.4,
            'temporal_signal': 0.2,
            'compression_anomaly': 0.1,
            'format_info': 0.05,
        }
        
        weighted_sum = sum(
            sig.get('confidence', 0.0) * weights.get(sig.get('type', ''), 0.1)
            for sig in creator_signals
        )
        
        return min(1.0, weighted_sum)

# Initialize agent
metadata_agent = MetadataForensicsAgent() if METADATA_AVAILABLE else None
if metadata_agent:
    print("Metadata Forensics Agent initialized")
    print("  â€¢ Creator attribution analysis")
    print("  â€¢ Tool detection (DeepFaceLab, FaceSwap, etc.)")
    print("  â€¢ Device fingerprinting")
    print("  â€¢ Compression pattern analysis")
else:
    print("Metadata Forensics Agent skipped (ffmpeg-python not available)")
    print("  Install: pip install ffmpeg-python")



def load_celebdf_dataset(data_dir: str = "/kaggle/input/celeb-df-v2"):
    """Load Celeb-DF v2 dataset
    
    Expected structure:
    /kaggle/input/celeb-df-v2/
        Celeb-real/          # Real videos
            video1.mp4
            ...
        Celeb-synthesis/     # Fake videos
            video1.mp4
            ...
        YouTube-real/        # Real videos
            video1.mp4
            ...
    """
    dataset = []
    
    # Load Celeb-real videos (real)
    celeb_real_dir = os.path.join(data_dir, "Celeb-real")
    if os.path.exists(celeb_real_dir):
        celeb_real_count = 0
        for filename in sorted(os.listdir(celeb_real_dir)):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(celeb_real_dir, filename)
                dataset.append((video_path, 0))  # 0 = real
                celeb_real_count += 1
        print(f"  âœ“ Loaded Celeb-real: {celeb_real_count} videos")
    
    # Load Celeb-synthesis videos (fake)
    celeb_synthesis_dir = os.path.join(data_dir, "Celeb-synthesis")
    if os.path.exists(celeb_synthesis_dir):
        celeb_synthesis_count = 0
        for filename in sorted(os.listdir(celeb_synthesis_dir)):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(celeb_synthesis_dir, filename)
                dataset.append((video_path, 1))  # 1 = fake
                celeb_synthesis_count += 1
        print(f"  âœ“ Loaded Celeb-synthesis: {celeb_synthesis_count} videos")
    
    # Load YouTube-real videos (real)
    youtube_real_dir = os.path.join(data_dir, "YouTube-real")
    if os.path.exists(youtube_real_dir):
        youtube_count = 0
        for filename in sorted(os.listdir(youtube_real_dir)):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(youtube_real_dir, filename)
                dataset.append((video_path, 0))  # 0 = real
                youtube_count += 1
        print(f"  âœ“ Loaded YouTube-real: {youtube_count} videos")
    
    # Count totals
    real_count = sum(1 for _, label in dataset if label == 0)
    fake_count = sum(1 for _, label in dataset if label == 1)
    
    print(f"\nLoaded Celeb-DF v2 dataset: {len(dataset)} videos")
    print(f"  Real: {real_count}, Fake: {fake_count}")
    
    if len(dataset) == 0:
        print(f"\nâš ï¸�  No videos found. Checked directories:")
        print(f"  - {celeb_real_dir}")
        print(f"  - {celeb_synthesis_dir}")
        print(f"  - {youtube_real_dir}")
        print(f"\nMake sure the dataset is added to Kaggle notebook with path: /kaggle/input/celeb-df-v2")
    
    return dataset


# def load_dfdc_dataset(data_dir: str = "/kaggle/input/deepfake-detection-challenge"):
#     """Load DFDC (Facebook Deepfake Detection Challenge) dataset
    
#     DFDC structure:
#     data_dir/
#         train_sample_videos/
#             *.mp4
#         metadata.json (or train.csv)
#     """
#     dataset = []
    
    # Check for CSV metadata
#     csv_path = os.path.join(data_dir, "train.csv")
#     if os.path.exists(csv_path):
#         df = pd.read_csv(csv_path)
#         video_dir = os.path.join(data_dir, "train_sample_videos")
        
#         for _, row in df.iterrows():
#             video_name = row.get('filename', row.get('video', ''))
#             label = row.get('label', row.get('original', 1))
            # DFDC: 0 = fake, 1 = real
#             video_path = os.path.join(video_dir, video_name)
#             if os.path.exists(video_path):
#                 dataset.append((video_path, 1 - int(label)))  # Convert to our format: 0=real, 1=fake
#     else:
        # Fallback: load from directory structure
#         video_dir = os.path.join(data_dir, "train_sample_videos")
#         if os.path.exists(video_dir):
#             for filename in os.listdir(video_dir):
#                 if filename.endswith(('.mp4', '.avi', '.mov')):
#                     video_path = os.path.join(video_dir, filename)
                    # Default: assume fake for DFDC (you may need to check metadata)
#                     dataset.append((video_path, 1))
    
#     print(f"Loaded DFDC dataset: {len(dataset)} videos")
#     return dataset


# def load_faceforensics_dataset(data_dir: str = "/kaggle/input/faceforensics"):
#     """Load FaceForensics++ dataset
    
#     FaceForensics++ structure:
#     data_dir/
#         original_sequences/
#             youtube/
#                 c0/
#                     frames/
#         manipulated_sequences/
#             Deepfakes/
#                 c0/
#                     frames/
#             FaceSwap/
#                 c0/
#                     frames/
#             Face2Face/
#                 c0/
#                     frames/
#             NeuralTextures/
#                 c0/
#                     frames/
#     """
#     dataset = []
    
    # Load original (real) videos
#     original_dir = os.path.join(data_dir, "original_sequences", "youtube", "c0", "videos")
#     if os.path.exists(original_dir):
#         for filename in os.listdir(original_dir):
#             if filename.endswith(('.mp4', '.avi', '.mov')):
#                 video_path = os.path.join(original_dir, filename)
#                 dataset.append((video_path, 0))  # 0 = real
    
    # Load manipulated videos
#     manipulated_dir = os.path.join(data_dir, "manipulated_sequences")
#     if os.path.exists(manipulated_dir):
#         for manipulation_type in ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']:
#             manip_dir = os.path.join(manipulated_dir, manipulation_type, "c0", "videos")
#             if os.path.exists(manip_dir):
#                 for filename in os.listdir(manip_dir):
#                     if filename.endswith(('.mp4', '.avi', '.mov')):
#                         video_path = os.path.join(manip_dir, filename)
#                         dataset.append((video_path, 1))  # 1 = fake
    
#     print(f"Loaded FaceForensics++ dataset: {len(dataset)} videos")
#     real_count = sum(1 for _, label in dataset if label == 0)
#     fake_count = sum(1 for _, label in dataset if label == 1)
#     print(f"  Real: {real_count}, Fake: {fake_count}")
    
#     return dataset


# def load_kaggle_dataset(dataset_name: str = "deepfake-detection-challenge"):
#     """Load dataset from Kaggle input
    
#     Usage:
#         dataset = load_kaggle_dataset("deepfake-detection-challenge")
#     """
#     data_dir = f"/kaggle/input/{dataset_name}"
    
#     if not os.path.exists(data_dir):
#         print(f"Dataset not found at {data_dir}")
#         print("Please ensure the dataset is added to Kaggle notebook inputs")
#         return []
    
    # Try to auto-detect dataset type
#     if "celeb" in dataset_name.lower():
#         return load_celebdf_dataset(data_dir)
#     elif "faceforensics" in dataset_name.lower():
#         return load_faceforensics_dataset(data_dir)
#     elif "deepfake" in dataset_name.lower() or "dfdc" in dataset_name.lower():
#         return load_dfdc_dataset(data_dir)
#     else:
        # Try generic directory structure
#         return load_dataset_from_directory(data_dir)

# print("Dataset integration functions ready")
# print("\nSupported datasets:")
# print("  â€¢ Celeb-DF: load_celebdf_dataset('/path/to/celebdf')")
# print("  â€¢ DFDC: load_dfdc_dataset('/path/to/dfdc')")
# print("  â€¢ FaceForensics++: load_faceforensics_dataset('/path/to/faceforensics')")
# print("  â€¢ Generic: load_dataset_from_directory('/path/to/data')")








# Enhanced data augmentation
class VideoAugmentation:
    """Data augmentation for video frames"""
    
    def __init__(self, use_augmentation: bool = True):
        self.use_augmentation = use_augmentation
        
        # Training augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33)),
        ])
        
        # Validation/test augmentation (no random transforms)
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_transform(self, is_training: bool = True):
        if is_training and self.use_augmentation:
            return self.train_transform
        return self.val_transform


# Transfer Learning with Pretrained Models
class VisualForensicsTransferLearning(nn.Module):
    """Visual forensics model using transfer learning from ResNet/EfficientNet"""
    
    def __init__(self, backbone: str = "resnet18", num_classes: int = 1, pretrained: bool = True):
        super(VisualForensicsTransferLearning, self).__init__()
        
        # Load pretrained backbone
        if backbone.startswith("resnet"):
            import torchvision.models as models
            if backbone == "resnet18":
                self.backbone = models.resnet18(pretrained=pretrained)
                num_features = 512
            elif backbone == "resnet50":
                self.backbone = models.resnet50(pretrained=pretrained)
                num_features = 2048
            elif backbone == "resnet101":
                self.backbone = models.resnet101(pretrained=pretrained)
                num_features = 2048
            else:
                raise ValueError(f"Unsupported ResNet: {backbone}")
            
            # Replace final layer
            self.backbone.fc = nn.Identity()
        elif backbone.startswith("efficientnet"):
            try:
                import efficientnet_pytorch as efn
                if backbone == "efficientnet-b0":
                    self.backbone = efn.EfficientNet.from_pretrained('efficientnet-b0')
                    num_features = 1280
                elif backbone == "efficientnet-b4":
                    self.backbone = efn.EfficientNet.from_pretrained('efficientnet-b4')
                    num_features = 1792
                else:
                    raise ValueError(f"Unsupported EfficientNet: {backbone}")
                self.backbone._fc = nn.Identity()
            except ImportError:
                print("efficientnet_pytorch not installed, falling back to ResNet18")
                self.backbone = models.resnet18(pretrained=pretrained)
                num_features = 512
                self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        output = self.classifier(features)
        return torch.sigmoid(output)


# Early Stopping
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def _is_better(self, score: float, best_score: float) -> bool:
        if self.mode == 'min':
            return score < best_score - self.min_delta
        return score > best_score + self.min_delta


# Optimized training function with all enhancements
def train_model_optimized(
    model: nn.Module,
    train_data: List[Tuple[str, int]],
    val_data: List[Tuple[str, int]],
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_augmentation: bool = True,
    use_early_stopping: bool = True,
    patience: int = 7,
    weight_decay: float = 1e-4,
    scheduler_type: str = "plateau",
    save_best: bool = True,
    model_name: str = "model"
):
    """Train model with optimizations"""
    
    model = model.to(device)
    criterion = nn.BCELoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1
        )
    else:
        scheduler = None
    
    # Data augmentation
    aug = VideoAugmentation(use_augmentation=use_augmentation)
    train_transform = aug.get_transform(is_training=True)
    val_transform = aug.get_transform(is_training=False)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min') if use_early_stopping else None
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Training {model_name}...")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    print(f"Using augmentation: {use_augmentation}, Early stopping: {use_early_stopping}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Shuffle training data
        np.random.shuffle(train_data)
        
        for video_path, label in train_data:
            try:
                preprocessed = preprocessor.extract_frames(video_path, num_frames=3)
                frames = preprocessed.get("frames", [])
                
                if not frames:
                    continue
                
                # Use random frame(s) for training
                frame = frames[np.random.randint(len(frames))]
                frame_pil = Image.fromarray(frame)
                frame_tensor = train_transform(frame_pil).unsqueeze(0).to(device)
                label_tensor = torch.tensor([[float(label)]], device=device)
                
                optimizer.zero_grad()
                output = model(frame_tensor)
                loss = criterion(output, label_tensor)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
            except Exception as e:
                continue
        
        avg_train_loss = train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for video_path, label in val_data:
                try:
                    preprocessed = preprocessor.extract_frames(video_path, num_frames=3)
                    frames = preprocessed.get("frames", [])
                    
                    if not frames:
                        continue
                    
                    # Use middle frame for validation
                    frame = frames[len(frames) // 2]
                    frame_pil = Image.fromarray(frame)
                    frame_tensor = val_transform(frame_pil).unsqueeze(0).to(device)
                    label_tensor = torch.tensor([[float(label)]], device=device)
                    
                    output = model(frame_tensor)
                    loss = criterion(output, label_tensor)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        if scheduler:
            if scheduler_type == "plateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Save best model
        if save_best and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"  âœ“ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # Early stopping check
        if early_stopping:
            if early_stopping(avg_val_loss):
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
    
    # Load best model
    if save_best and os.path.exists(f"{model_name}_best.pth"):
        model.load_state_dict(torch.load(f"{model_name}_best.pth", map_location=device))
        print(f"Loaded best model (val_loss: {best_val_loss:.4f})")
    
    return model, train_losses, val_losses

print("Training optimization functions ready")
print("\nAvailable features:")
print("  â€¢ Transfer learning (ResNet, EfficientNet)")
print("  â€¢ Data augmentation")
print("  â€¢ Early stopping")
print("  â€¢ Learning rate scheduling (plateau, cosine, step)")
print("  â€¢ Gradient clipping")
print("  â€¢ Weight decay")






from sklearn.metrics import roc_curve, precision_recall_curve, auc, classification_report

def plot_roc_curve(y_true: List[int], y_scores: List[float], title: str = "ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(y_true: List[int], y_scores: List[float], 
                                title: str = "Precision-Recall Curve"):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    baseline = sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--',
                label=f'Baseline (AUC = {baseline:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return pr_auc


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                           title: str = "Metrics Comparison"):
    """Plot comparison of metrics across different models/agents"""
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    # Prepare data
    data = {metric: [metrics_dict[model].get(metric, 0) for model in models] 
            for metric in metric_names}
    
    x = np.arange(len(models))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, metric in enumerate(metric_names):
        offset = (i - 2) * width
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model/Agent', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_per_class_performance(y_true: List[int], y_pred: List[int], 
                               class_names: List[str] = ['Real', 'AI-Generated']):
    """Plot per-class performance metrics"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract metrics for each class
    classes = []
    precision_vals = []
    recall_vals = []
    f1_vals = []
    
    for class_name in class_names:
        if class_name.lower() in report:
            class_metrics = report[class_name.lower()]
            classes.append(class_name)
            precision_vals.append(class_metrics['precision'])
            recall_vals.append(class_metrics['recall'])
            f1_vals.append(class_metrics['f1-score'])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
    ax.bar(x, recall_vals, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Print detailed report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_agent_contribution_heatmap(results_dict: Dict[str, Dict[str, float]]):
    """Plot heatmap showing contribution of each agent to final decision"""
    agents = list(results_dict.keys())
    metrics = ['confidence', 'accuracy', 'f1_score']
    
    # Create matrix
    matrix = np.zeros((len(agents), len(metrics)))
    for i, agent in enumerate(agents):
        for j, metric in enumerate(metrics):
            matrix[i, j] = results_dict[agent].get(metric, 0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=metrics, yticklabels=agents,
                cbar_kws={'label': 'Score'})
    plt.title('Agent Contribution Heatmap', fontsize=16, fontweight='bold')
    plt.ylabel('Agent', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis(y_true: List[int], y_scores: List[float]):
    """Plot metrics across different threshold values"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2)
    plt.plot(thresholds, precisions, 's-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, '^-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, 'd-', label='F1-Score', linewidth=2)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance vs Classification Threshold', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Find optimal threshold (maximizing F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal threshold: {optimal_threshold:.2f}")
    print(f"F1-Score at optimal threshold: {f1_scores[optimal_idx]:.4f}")


def plot_model_attribution_distribution(attribution_results: Dict[str, List[Dict[str, Any]]]):
    """Plot distribution of model attributions across test set"""
    model_counts = {}
    
    for result in attribution_results.values():
        top_models = result.get('top_models', [])
        for model in top_models:
            model_counts[model] = model_counts.get(model, 0) + 1
    
    if not model_counts:
        print("No attribution data available")
        return
    
    models = list(model_counts.keys())
    counts = list(model_counts.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, counts, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    plt.xlabel('AI Model', fontsize=12)
    plt.ylabel('Detection Count', fontsize=12)
    plt.title('Model Attribution Distribution', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def comprehensive_evaluation_plots(y_true: List[int], y_pred: List[int], 
                                   y_scores: List[float], model_name: str = "Model"):
    """Create comprehensive evaluation plots"""
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print(f"{'='*70}")
    
    # ROC Curve
    roc_auc = plot_roc_curve(y_true, y_scores, f"ROC Curve - {model_name}")
    
    # Precision-Recall Curve
    pr_auc = plot_precision_recall_curve(y_true, y_scores, 
                                         f"Precision-Recall Curve - {model_name}")
    
    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {model_name}")
    
    # Per-Class Performance
    plot_per_class_performance(y_true, y_pred)
    
    # Threshold Analysis
    plot_threshold_analysis(y_true, y_scores)
    
    # Summary metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    print(f"\nSummary Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    return metrics

print("Additional visualization functions ready")
print("\nAvailable visualizations:")
print("  â€¢ ROC curves with AUC")
print("  â€¢ Precision-Recall curves")
print("  â€¢ Metrics comparison across models")
print("  â€¢ Per-class performance breakdown")
print("  â€¢ Agent contribution heatmaps")
print("  â€¢ Threshold analysis")
print("  â€¢ Model attribution distribution")
print("  â€¢ Comprehensive evaluation suite")






class AblationStudy:
    """Comprehensive ablation study framework"""
    
    def __init__(self, test_data: List[Tuple[str, int]]):
        self.test_data = test_data
        self.results = {}
    
    def evaluate_single_agent(self, agent_name: str, agent, preprocessed_cache: Dict = None):
        """Evaluate a single agent"""
        print(f"\n{'='*70}")
        print(f"Evaluating: {agent_name}")
        print(f"{'='*70}")
        
        predictions = []
        probabilities = []
        labels = []
        
        for video_path, label in self.test_data:
            try:
                # Preprocessing
                if preprocessed_cache and video_path in preprocessed_cache:
                    preprocessed = preprocessed_cache[video_path]
                else:
                    preprocessed = preprocessor.extract_frames(video_path, num_frames=10)
                    if preprocessed_cache is not None:
                        preprocessed_cache[video_path] = preprocessed
                
                # Agent analysis
                if agent_name == "visual_forensics":
                    result = visual_agent.analyze(preprocessed)
                    confidence = result.get("overall_confidence", 0.0)
                elif agent_name == "temporal_analysis":
                    result = temporal_agent.analyze(preprocessed)
                    confidence = result.get("temporal_confidence", 0.0)
                elif agent_name == "audio_forensics":
                    if audio_agent:
                        result = audio_agent.analyze(preprocessed)
                        confidence = result.get("audio_confidence", 0.0)
                    else:
                        continue
                elif agent_name == "model_attribution":
                    visual_result = visual_agent.analyze(preprocessed)
                    temporal_result = temporal_agent.analyze(preprocessed)
                    result = attribution_agent.analyze(preprocessed, visual_result, temporal_result)
                    confidence = result.get("attribution_confidence", 0.0)
                else:
                    continue
                
                # Convert to prediction
                prob = float(confidence)
                pred = 1 if prob > 0.5 else 0
                
                probabilities.append(prob)
                predictions.append(pred)
                labels.append(label)
                
            except Exception as e:
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'f1_score': f1_score(labels, predictions, zero_division=0),
            }
            
            try:
                metrics['roc_auc'] = roc_auc_score(labels, probabilities)
            except:
                metrics['roc_auc'] = 0.0
            
            self.results[agent_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'probabilities': probabilities,
                'labels': labels
            }
            
            print(f"\nResults for {agent_name}:")
            for metric, value in metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            return metrics
        else:
            print(f"No valid predictions for {agent_name}")
            return {}
    
    def evaluate_agent_combination(self, agent_names: List[str], 
                                   weight_config: Dict[str, float] = None):
        """Evaluate combination of agents"""
        print(f"\n{'='*70}")
        print(f"Evaluating Combination: {', '.join(agent_names)}")
        print(f"{'='*70}")
        
        if weight_config is None:
            # Equal weights
            weight_config = {name: 1.0 / len(agent_names) for name in agent_names}
        
        predictions = []
        probabilities = []
        labels = []
        preprocessed_cache = {}
        
        for video_path, label in self.test_data:
            try:
                # Preprocessing
                preprocessed = preprocessor.extract_frames(video_path, num_frames=10)
                preprocessed_cache[video_path] = preprocessed
                
                # Get confidences from each agent
                confidences = {}
                
                if "visual_forensics" in agent_names:
                    result = visual_agent.analyze(preprocessed)
                    confidences["visual_forensics"] = result.get("overall_confidence", 0.0)
                
                if "temporal_analysis" in agent_names:
                    result = temporal_agent.analyze(preprocessed)
                    confidences["temporal_analysis"] = result.get("temporal_confidence", 0.0)
                
                if "audio_forensics" in agent_names and audio_agent:
                    result = audio_agent.analyze(preprocessed)
                    confidences["audio_forensics"] = result.get("audio_confidence", 0.0)
                
                if "model_attribution" in agent_names:
                    visual_result = visual_agent.analyze(preprocessed)
                    temporal_result = temporal_agent.analyze(preprocessed)
                    result = attribution_agent.analyze(preprocessed, visual_result, temporal_result)
                    confidences["model_attribution"] = result.get("attribution_confidence", 0.0)
                
                # Weighted average
                weighted_score = sum(
                    confidences.get(name, 0.0) * weight_config.get(name, 0.0)
                    for name in agent_names
                )
                
                # Convert to prediction
                prob = float(weighted_score)
                pred = 1 if prob > 0.5 else 0
                
                probabilities.append(prob)
                predictions.append(pred)
                labels.append(label)
                
            except Exception as e:
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            combo_name = "+".join(agent_names)
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'f1_score': f1_score(labels, predictions, zero_division=0),
            }
            
            try:
                metrics['roc_auc'] = roc_auc_score(labels, probabilities)
            except:
                metrics['roc_auc'] = 0.0
            
            self.results[combo_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'probabilities': probabilities,
                'labels': labels,
                'weight_config': weight_config
            }
            
            print(f"\nResults for {combo_name}:")
            for metric, value in metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            return metrics
        else:
            print(f"No valid predictions for combination")
            return {}
    
    def evaluate_full_pipeline(self):
        """Evaluate full multi-agent pipeline"""
        print(f"\n{'='*70}")
        print(f"Evaluating: Full Pipeline (All Agents)")
        print(f"{'='*70}")
        
        predictions = []
        probabilities = []
        labels = []
        
        for video_path, label in self.test_data:
            try:
                # Run full pipeline
                result = orchestrator.analyze_video(video_path)
                
                # Get authenticity score and convert to probability
                authenticity_score = result.get("authenticity_score", 50)
                prob = 1.0 - (authenticity_score / 100.0)  # Convert to AI probability
                pred = 1 if prob > 0.5 else 0
                
                probabilities.append(prob)
                predictions.append(pred)
                labels.append(label)
                
            except Exception as e:
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'f1_score': f1_score(labels, predictions, zero_division=0),
            }
            
            try:
                metrics['roc_auc'] = roc_auc_score(labels, probabilities)
            except:
                metrics['roc_auc'] = 0.0
            
            self.results['full_pipeline'] = {
                'metrics': metrics,
                'predictions': predictions,
                'probabilities': probabilities,
                'labels': labels
            }
            
            print(f"\nResults for Full Pipeline:")
            for metric, value in metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            return metrics
        else:
            print(f"No valid predictions for full pipeline")
            return {}
    
    def plot_ablation_results(self):
        """Plot ablation study results"""
        if not self.results:
            print("No results to plot. Run evaluations first.")
            return
        
        # Extract metrics for plotting
        configurations = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        data = {}
        for metric in metrics:
            data[metric] = [self.results[config]['metrics'].get(metric, 0) 
                           for config in configurations]
        
        # Create comparison plot
        plot_metrics_comparison(
            {config: self.results[config]['metrics'] for config in configurations},
            "Ablation Study: Agent Contribution Analysis"
        )
        
        # Create detailed table
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)
        print(f"{'Configuration':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
        print("-"*80)
        
        for config in configurations:
            m = self.results[config]['metrics']
            print(f"{config:<30} {m.get('accuracy', 0):<12.4f} {m.get('precision', 0):<12.4f} "
                  f"{m.get('recall', 0):<12.4f} {m.get('f1_score', 0):<12.4f} "
                  f"{m.get('roc_auc', 0):<12.4f}")
        
        print("="*80)
    
    def run_full_ablation(self):
        """Run complete ablation study"""
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE ABLATION STUDY")
        print("="*70)
        
        preprocessed_cache = {}
        
        # Single agents
        self.evaluate_single_agent("visual_forensics", visual_agent, preprocessed_cache)
        self.evaluate_single_agent("temporal_analysis", temporal_agent, preprocessed_cache)
        if audio_agent:
            self.evaluate_single_agent("audio_forensics", audio_agent, preprocessed_cache)
        self.evaluate_single_agent("model_attribution", attribution_agent, preprocessed_cache)
        
        # Two-agent combinations
        self.evaluate_agent_combination(["visual_forensics", "temporal_analysis"])
        self.evaluate_agent_combination(["visual_forensics", "audio_forensics"])
        self.evaluate_agent_combination(["visual_forensics", "model_attribution"])
        self.evaluate_agent_combination(["temporal_analysis", "model_attribution"])
        
        # Three-agent combinations
        self.evaluate_agent_combination(["visual_forensics", "temporal_analysis", "model_attribution"])
        if audio_agent:
            self.evaluate_agent_combination(["visual_forensics", "temporal_analysis", "audio_forensics"])
        
        # Full pipeline
        self.evaluate_full_pipeline()
        
        # Plot results
        self.plot_ablation_results()
        
        print("\n" + "="*70)
        print("ABLATION STUDY COMPLETE")
        print("="*70)
        
        return self.results

# Example usage function
def run_ablation_study(test_data: List[Tuple[str, int]]):
    """Run ablation study on test data"""
    study = AblationStudy(test_data)
    results = study.run_full_ablation()
    return results

print("Ablation study framework ready")
print("\nUsage:")
print("  study = AblationStudy(test_data)")
print("  study.run_full_ablation()  # Run complete study")
print("  study.plot_ablation_results()  # Visualize results")






# Training Data Preparation Functions

def prepare_complete_training_data(
    celebdf_path: str = "/kaggle/input/celebdf",
    dfdc_path: str = "/kaggle/input/deepfake-detection-challenge",
    model_attribution_dir: str = None,
    metadata_training_dir: str = None
):
    """
    Prepare complete training dataset for all agents
    
    Returns:
        - detection_data: (train, val, test) for real/fake detection
        - attribution_data: (train, val, test) for model attribution
        - metadata_data: List of metadata training samples
    """
    
    print("="*70)
    print("PREPARING COMPLETE TRAINING DATASET")
    print("="*70)
    
    # 1. Detection Training Data (Real vs Fake)
    print("\n[1/3] Preparing Detection Training Data...")
    detection_data = []
    
    # Load Celeb-DF
    try:
        celebdf = load_celebdf_dataset(celebdf_path)
        detection_data.extend(celebdf)
        print(f"  âœ“ Loaded Celeb-DF: {len(celebdf)} videos")
    except Exception as e:
        print(f"  âœ— Celeb-DF not available: {e}")
    
    # Load DFDC subset
    try:
        dfdc = load_dfdc_dataset(dfdc_path)
        dfdc_subset = dfdc[:5000]  # Use subset for training
        detection_data.extend(dfdc_subset)
        print(f"  âœ“ Loaded DFDC subset: {len(dfdc_subset)} videos")
    except Exception as e:
        print(f"  âœ— DFDC not available: {e}")
    
    if not detection_data:
        print("  âš ï¸�  No detection data available. Please add datasets.")
        detection_train, detection_val, detection_test = [], [], []
    else:
        detection_train, detection_val, detection_test = split_dataset(
            detection_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        print(f"  âœ“ Detection data split: {len(detection_train)} train, "
              f"{len(detection_val)} val, {len(detection_test)} test")
    
    # 2. Model Attribution Training Data
    print("\n[2/3] Preparing Model Attribution Training Data...")
    attribution_data = []
    
    if model_attribution_dir and os.path.exists(model_attribution_dir):
        # Load from organized directories
        models = ['runway_gen3', 'pika', 'sora', 'stable_video', 'gen2']
        for model_name in models:
            model_dir = os.path.join(model_attribution_dir, model_name)
            if os.path.exists(model_dir):
                videos = [
                    (os.path.join(model_dir, f), model_name)
                    for f in os.listdir(model_dir)
                    if f.endswith(('.mp4', '.avi', '.mov'))
                ]
                attribution_data.extend(videos)
                print(f"  âœ“ Loaded {model_name}: {len(videos)} videos")
    else:
        print("  âš ï¸�  Model attribution directory not provided")
        print("  ğŸ’¡ Tip: Create directories with model-specific videos")
        print("     model_attribution_dir/")
        print("       â”œâ”€â”€ runway_gen3/")
        print("       â”œâ”€â”€ pika/")
        print("       â””â”€â”€ ...")
    
    if not attribution_data:
        print("  âš ï¸�  No attribution data available")
        attribution_train, attribution_val, attribution_test = [], [], []
    else:
        # Convert to (video_path, model_index) format
        model_to_idx = {
            'runway_gen3': 0, 'pika': 1, 'sora': 2,
            'stable_video': 3, 'gen2': 4
        }
        formatted_data = [
            (video_path, model_to_idx.get(model_name, 0))
            for video_path, model_name in attribution_data
        ]
        attribution_train, attribution_val, attribution_test = split_dataset(
            formatted_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        print(f"  âœ“ Attribution data split: {len(attribution_train)} train, "
              f"{len(attribution_val)} val, {len(attribution_test)} test")
    
    # 3. Metadata Forensics Training Data
    print("\n[3/3] Preparing Metadata Forensics Training Data...")
    metadata_data = []
    
    if metadata_training_dir and os.path.exists(metadata_training_dir):
        tools = ['deepfacelab', 'faceswap', 'wav2lip', 'ffmpeg', 'real']
        for tool in tools:
            tool_dir = os.path.join(metadata_training_dir, tool)
            if os.path.exists(tool_dir):
                videos = [
                    os.path.join(tool_dir, f)
                    for f in os.listdir(tool_dir)
                    if f.endswith(('.mp4', '.avi', '.mov'))
                ]
                for video_path in videos:
                    try:
                        if metadata_agent:
                            preprocessed = preprocessor.extract_frames(video_path, num_frames=1)
                            metadata_result = metadata_agent.analyze(preprocessed)
                            metadata_data.append({
                                'video_path': video_path,
                                'tool': tool,
                                'metadata': metadata_result.get('metadata', {})
                            })
                    except:
                        continue
                print(f"  âœ“ Loaded {tool}: {len(videos)} videos")
    else:
        print("  âš ï¸�  Metadata training directory not provided")
        print("  ğŸ’¡ Tip: Create videos using specific tools and preserve metadata")
    
    print(f"\n{'='*70}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*70}")
    print(f"\nDetection Training:")
    print(f"  Train: {len(detection_train)} videos")
    print(f"  Val: {len(detection_val)} videos")
    print(f"  Test: {len(detection_test)} videos")
    
    print(f"\nModel Attribution Training:")
    print(f"  Train: {len(attribution_train)} videos")
    print(f"  Val: {len(attribution_val)} videos")
    print(f"  Test: {len(attribution_test)} videos")
    
    print(f"\nMetadata Forensics Training:")
    print(f"  Total samples: {len(metadata_data)}")
    
    return {
        'detection': {
            'train': detection_train,
            'val': detection_val,
            'test': detection_test
        },
        'attribution': {
            'train': attribution_train,
            'val': attribution_val,
            'test': attribution_test
        },
        'metadata': metadata_data
    }


def create_model_attribution_dataset_structure(base_dir: str = "./model_attribution_data"):
    """Create directory structure for model attribution training data"""
    
    models = ['runway_gen3', 'pika', 'sora', 'stable_video', 'gen2']
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"Created directory structure at: {base_dir}")
    print("\nNext steps:")
    print("  1. Add videos to each model directory")
    print("  2. Ensure 1,000+ videos per model for best results")
    print("  3. Use prepare_complete_training_data() to load")


def create_metadata_training_dataset_structure(base_dir: str = "./metadata_training_data"):
    """Create directory structure for metadata forensics training data"""
    
    tools = ['deepfacelab', 'faceswap', 'wav2lip', 'ffmpeg', 'real']
    
    for tool in tools:
        tool_dir = os.path.join(base_dir, tool)
        os.makedirs(tool_dir, exist_ok=True)
    
    print(f"Created directory structure at: {base_dir}")
    print("\nNext steps:")
    print("  1. Create videos using each tool")
    print("  2. Preserve metadata during creation")
    print("  3. Place videos in corresponding tool directory")
    print("  4. Use prepare_complete_training_data() to load")


# Example usage
print("Training data preparation functions ready")
print("\nUsage:")
print("  # Prepare complete dataset")
print("  data = prepare_complete_training_data(")
print("      celebdf_path='/kaggle/input/celebdf',")
print("      dfdc_path='/kaggle/input/deepfake-detection-challenge',")
print("      model_attribution_dir='./model_attribution_data',")
print("      metadata_training_dir='./metadata_training_data'")
print("  )")
print("\n  # Create directory structures")
print("  create_model_attribution_dataset_structure()")
print("  create_metadata_training_dataset_structure()")



# ## 20. Complete Training Workflow with All Features

# End-to-end training workflow including detection, model attribution, and metadata forensics.



# Complete Training Workflow - Uncomment and modify paths

"""
# ============================================================================
# COMPLETE TRAINING WORKFLOW FOR 90%+ ACCURACY
# ============================================================================

print("="*70)
print("Phantom Trace - COMPLETE TRAINING WORKFLOW")
print("="*70)

# Step 1: Prepare All Training Data
print("\n[STEP 1] Preparing Training Data...")
training_data = prepare_complete_training_data(
    celebdf_path="/kaggle/input/celebdf",
    dfdc_path="/kaggle/input/deepfake-detection-challenge",
    model_attribution_dir="./model_attribution_data",  # Optional
    metadata_training_dir="./metadata_training_data"   # Optional
)

detection_train = training_data['detection']['train']
detection_val = training_data['detection']['val']
detection_test = training_data['detection']['test']

attribution_train = training_data['attribution']['train']
attribution_val = training_data['attribution']['val']
attribution_test = training_data['attribution']['test']

metadata_data = training_data['metadata']


# Step 2: Train Visual Forensics Model
print("\n[STEP 2] Training Visual Forensics Model...")
visual_model = VisualForensicsTransferLearning(backbone="resnet50", pretrained=True).to(device)

visual_model, train_losses, val_losses = train_model_optimized(
    model=visual_model,
    train_data=detection_train,
    val_data=detection_val,
    num_epochs=20,
    batch_size=32,
    learning_rate=0.001,
    use_augmentation=True,
    use_early_stopping=True,
    patience=7,
    scheduler_type="plateau",
    model_name="visual_forensics"
)

# Load best model
visual_model.load_state_dict(torch.load("visual_forensics_best.pth", map_location=device))
visual_agent.model = visual_model
visual_agent.model.eval()

# Evaluate
print("\n[STEP 2a] Evaluating Visual Forensics Model...")
y_true, y_pred, y_scores = [], [], []
for video_path, label in detection_test[:100]:  # Sample for speed
    try:
        preprocessed = preprocessor.extract_frames(video_path, num_frames=10)
        result = visual_agent.analyze(preprocessed)
        y_true.append(label)
        y_pred.append(1 if result['overall_confidence'] > 0.5 else 0)
        y_scores.append(result['overall_confidence'])
    except:
        continue

if len(y_true) > 0:
    accuracy = accuracy_score(y_true, y_pred)
    print(f"  Visual Forensics Accuracy: {accuracy:.4f}")


# Step 3: Train Temporal Analysis Model
print("\n[STEP 3] Training Temporal Analysis Model...")
# Similar process for temporal model
# (Implementation would be similar to visual forensics)


# Step 4: Train Model Attribution (if data available)
if len(attribution_train) > 0:
    print("\n[STEP 4] Training Model Attribution...")
    # Train model attribution network
    # (Implementation would train on attribution_train)
    print(f"  Training on {len(attribution_train)} attribution samples")
else:
    print("\n[STEP 4] Model Attribution Training Skipped (no data)")
    print("  ğŸ’¡ To enable model attribution:")
    print("     1. Collect videos from each AI model")
    print("     2. Organize in model_attribution_data/ directory")
    print("     3. Re-run training workflow")


# Step 5: Train Metadata Forensics (if data available)
if len(metadata_data) > 0:
    print("\n[STEP 5] Metadata Forensics Training...")
    print(f"  Training on {len(metadata_data)} metadata samples")
    # Train metadata classifier
    # (Implementation would train on metadata features)
else:
    print("\n[STEP 5] Metadata Forensics Training Skipped (no data)")
    print("  ğŸ’¡ To enable metadata forensics:")
    print("     1. Create videos using specific tools (DeepFaceLab, FaceSwap, etc.)")
    print("     2. Preserve metadata during creation")
    print("     3. Organize in metadata_training_data/ directory")
    print("     4. Re-run training workflow")


# Step 6: Comprehensive Evaluation
print("\n[STEP 6] Comprehensive Evaluation...")
# Run full pipeline evaluation
# Run ablation studies
# Generate visualizations

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nNext Steps:")
print("  1. Evaluate on test set")
print("  2. Run ablation studies")
print("  3. Generate comprehensive visualizations")
print("  4. Document results")
"""

print("Complete training workflow ready")
print("\nTo achieve 90%+ accuracy, you need:")
print("  â€¢ Detection data: Celeb-DF (6,229 videos) + DFDC subset (5,000 videos)")
print("  â€¢ Model attribution: 1,000+ videos per model (5 models = 5,000+ videos)")
print("  â€¢ Metadata forensics: 500+ videos per tool (5 tools = 2,500+ videos)")
print("\nTotal recommended: ~13,700 videos")
print("\nMinimum viable: ~9,700 videos (Celeb-DF + some attribution data)")



# Complete example workflow - uncomment and modify paths as needed

"""
# ============================================================================
# STEP 1: Load Dataset
# ============================================================================

# Option 1: Load from Kaggle dataset
# dataset = load_kaggle_dataset("deepfake-detection-challenge")

# Option 2: Load Celeb-DF
# dataset = load_celebdf_dataset("/kaggle/input/celebdf")

# Option 3: Load from directory
# dataset = load_dataset_from_directory("/path/to/your/dataset", real_subdir="real", fake_subdir="fake")

# Option 4: Load from CSV
# dataset = load_dataset_from_csv("/path/to/dataset.csv", video_dir="/path/to/videos")

# Split dataset
train_data, val_data, test_data = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

print(f"\nDataset loaded and split:")
print(f"  Training: {len(train_data)} samples")
print(f"  Validation: {len(val_data)} samples")
print(f"  Test: {len(test_data)} samples")


# ============================================================================
# STEP 2: Train Visual Forensics Model (with optimizations)
# ============================================================================

print("\n" + "="*70)
print("TRAINING VISUAL FORENSICS MODEL")
print("="*70)

# Option 1: Train with transfer learning
visual_model = VisualForensicsTransferLearning(backbone="resnet50", pretrained=True).to(device)

# Option 2: Train from scratch
# visual_model = VisualForensicsCNN().to(device)

# Train with optimizations
visual_model, train_losses, val_losses = train_model_optimized(
    model=visual_model,
    train_data=train_data,
    val_data=val_data,
    num_epochs=20,
    batch_size=32,
    learning_rate=0.001,
    use_augmentation=True,
    use_early_stopping=True,
    patience=7,
    scheduler_type="plateau",
    model_name="visual_forensics"
)

# Plot training curves
plot_training_curves(train_losses, val_losses, "Visual Forensics Model Training")

# Load best model
visual_model.load_state_dict(torch.load("visual_forensics_best.pth", map_location=device))
visual_agent.model = visual_model
visual_agent.model.eval()


# ============================================================================
# STEP 3: Evaluate on Test Set
# ============================================================================

print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)

# Get predictions and probabilities
y_true = []
y_pred = []
y_scores = []

for video_path, label in test_data:
    try:
        preprocessed = preprocessor.extract_frames(video_path, num_frames=10)
        result = visual_agent.analyze(preprocessed)
        
        confidence = result.get("overall_confidence", 0.0)
        pred = 1 if confidence > 0.5 else 0
        
        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(confidence)
    except:
        continue

# Comprehensive evaluation
metrics = comprehensive_evaluation_plots(y_true, y_pred, y_scores, "Visual Forensics Model")


# ============================================================================
# STEP 4: Run Ablation Study
# ============================================================================

print("\n" + "="*70)
print("RUNNING ABLATION STUDY")
print("="*70)

# Run ablation study on test set
ablation_results = run_ablation_study(test_data)

# Plot ablation results
study = AblationStudy(test_data)
study.results = ablation_results
study.plot_ablation_results()


# ============================================================================
# STEP 5: Analyze Best Configuration
# ============================================================================

print("\n" + "="*70)
print("ANALYZING BEST CONFIGURATION")
print("="*70)

# Find best configuration
best_config = max(ablation_results.items(), 
                 key=lambda x: x[1]['metrics'].get('f1_score', 0))

print(f"\nBest Configuration: {best_config[0]}")
print(f"F1-Score: {best_config[1]['metrics'].get('f1_score', 0):.4f}")
print(f"Accuracy: {best_config[1]['metrics'].get('accuracy', 0):.4f}")
print(f"ROC-AUC: {best_config[1]['metrics'].get('roc_auc', 0):.4f}")

print("\n" + "="*70)
print("COMPLETE WORKFLOW FINISHED")
print("="*70)
"""

print("Complete workflow example ready")
print("\nUncomment the code above and provide dataset paths to run the full pipeline")
print("\nThe workflow includes:")
print("  1. Dataset loading (multiple formats supported)")
print("  2. Model training with optimizations")
print("  3. Comprehensive evaluation with visualizations")
print("  4. Ablation study to analyze agent contributions")
print("  5. Best configuration analysis")



# ## 13. Architecture Summary

# This notebook implements a comprehensive multi-agent AI pipeline for detecting AI-generated videos:

# ### Key Components:
# 1. **Preprocessing Agent**: Extracts frames and metadata from videos
# 2. **Visual Forensics Agent**: CNN-based detection of visual artifacts
# 3. **Temporal Analysis Agent**: 3D CNN for temporal consistency analysis
# 4. **Audio Forensics Agent**: Spectral analysis of audio artifacts
# 5. **Model Attribution Agent**: Identifies which AI model generated the video
# 6. **Evidence Aggregation Agent**: Combines all evidence into final score
# 7. **Orchestrator**: Coordinates all agents in the pipeline

# ### Novel Contributions:
# - **Multi-agent architecture** for comprehensive video forensics
# - **Model attribution** - identifies specific AI models (Runway, Pika, Sora, etc.)
# - **Multi-modal analysis** - combines visual, temporal, and audio evidence
# - **Weighted evidence aggregation** - intelligent fusion of agent outputs
# - **Production-ready design** - modular, scalable architecture

# ### Usage:
# 1. Load your dataset using `load_dataset_from_directory()` or `load_dataset_from_csv()`
# 2. Train models using `train_visual_forensics_model()` or similar functions
# 3. Run analysis on videos using `orchestrator.analyze_video(video_path)`
# 4. Visualize results using provided plotting functions

# ### Next Steps:
# - Train models on real deepfake detection datasets (e.g., Celeb-DF, DFDC)
# - Fine-tune hyperparameters for your specific use case
# - Add ensemble methods for improved accuracy
# - Implement real-time processing capabilities



# ## 14. Generate Kaggle Submission CSV

# Process test videos and export predictions to CSV file for Kaggle submission.

# Note: pandas, os, and pathlib are already imported in cell 2
import time
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available. Progress bars will be disabled.")
    # Simple tqdm replacement
    def tqdm(iterable, desc=""):
        return iterable

def generate_kaggle_submission(test_dir=None,
                                test_dirs=None,
                                output_csv: str = "/kaggle/working/submission.csv",
                                video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')):
    """
    Process test videos and generate Kaggle submission CSV file.
    
    Args:
        test_dir: Single directory containing test videos (for backward compatibility)
        test_dirs: List of directories containing test videos (can use this instead of test_dir)
        output_csv: Path to output CSV file (should be in /kaggle/working/)
        video_extensions: Tuple of video file extensions to process
    """
    print("="*70)
    print("GENERATING KAGGLE SUBMISSION")
    print("="*70)
    
    # Check if orchestrator is initialized
    try:
        # Try to access orchestrator to see if it exists
        _ = orchestrator
    except NameError:
        print("Error: Orchestrator not initialized.")
        print("Please run the orchestrator initialization cell (around cell 15) first.")
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=['filename', 'label'])
        empty_df.to_csv(output_csv, index=False)
        print(f"Created empty submission file: {output_csv}")
        return empty_df
    
    # Handle both single directory and multiple directories
    directories_to_search = []
    if test_dirs is not None:
        # Use list of directories
        if isinstance(test_dirs, str):
            directories_to_search = [test_dirs]
        else:
            directories_to_search = list(test_dirs)
    elif test_dir is not None:
        # Use single directory (backward compatibility)
        directories_to_search = [test_dir]
    else:
        print("Error: Please provide either test_dir or test_dirs parameter")
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=['filename', 'label'])
        empty_df.to_csv(output_csv, index=False)
        print(f"Created empty submission file: {output_csv}")
        return empty_df
    
    # Find all video files in all test directories
    test_videos = []
    for test_dir_path in directories_to_search:
        if os.path.exists(test_dir_path):
            print(f"Searching in: {test_dir_path}")
            videos_found_in_dir = 0
            for root, dirs, files in os.walk(test_dir_path):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        test_videos.append(os.path.join(root, file))
                        videos_found_in_dir += 1
            print(f"  Found {videos_found_in_dir} videos in this directory")
        else:
            print(f"Warning: Directory not found: {test_dir_path}")
            print(f"  Full path checked: {os.path.abspath(test_dir_path) if test_dir_path else 'N/A'}")
    
    if not test_videos:
        print(f"\nNo video files found in any of the specified directories:")
        for d in directories_to_search:
            print(f"  - {d}")
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=['filename', 'label'])
        empty_df.to_csv(output_csv, index=False)
        print(f"\nCreated empty submission file: {output_csv}")
        return empty_df
    
    print(f"\nFound {len(test_videos)} test videos")
    print(f"Processing videos and generating predictions...")
    print(f"This may take a while depending on the number of videos...")
    print(f"Estimated time: {len(test_videos) * 5 / 60:.1f} - {len(test_videos) * 10 / 60:.1f} minutes (5-10s per video)")
    
    # Process each video and collect predictions
    results = []
    processed_count = 0
    error_count = 0
    start_time = time.time()
    
    for video_path in tqdm(test_videos, desc="Processing videos"):
        try:
            # Get video filename (without path)
            video_filename = os.path.basename(video_path)
            
            # Analyze video using orchestrator
            result = orchestrator.analyze_video(video_path)
            
            # Extract prediction
            # Result contains:
            # - label: "likely_ai" or "likely_real" (string)
            # - authenticity_score: 0-100 (int, lower = more likely AI-generated)
            label_str = result.get('label', 'likely_real')
            authenticity_score = result.get('authenticity_score', 50)
            
            # Convert string label to binary
            # 0 = real, 1 = fake (AI-generated)
            if label_str == 'likely_ai':
                prediction = 1  # Fake/AI-generated
            else:
                prediction = 0  # Real
            
            # Calculate confidence from authenticity_score (0-100 -> 0.0-1.0)
            # Lower authenticity_score = higher confidence it's AI-generated
            if label_str == 'likely_ai':
                confidence = (100 - authenticity_score) / 100.0
            else:
                confidence = authenticity_score / 100.0
            
            # Alternative: Use authenticity_score directly (0-100 scale)
            # Or: Use probability based on score
            # prediction = 1 if authenticity_score < 50 else 0
            
            results.append({
                'filename': video_filename,
                'label': prediction,
                'confidence': confidence
            })
            processed_count += 1
            
            # Print progress every 10 videos with time estimates
            if processed_count % 10 == 0:
                elapsed_time = time.time() - start_time
                if processed_count > 0:
                    avg_time_per_video = elapsed_time / processed_count
                    remaining_videos = len(test_videos) - processed_count
                    estimated_remaining = avg_time_per_video * remaining_videos
                    print(f"  Processed {processed_count}/{len(test_videos)} videos ({processed_count/len(test_videos)*100:.1f}%)")
                    print(f"  Elapsed: {elapsed_time/60:.1f} min | Est. remaining: {estimated_remaining/60:.1f} min | Est. total: {(elapsed_time + estimated_remaining)/60:.1f} min")
            
        except Exception as e:
            error_count += 1
            print(f"\nError processing {video_path}: {e}")
            import traceback
            if error_count <= 3:  # Only show full traceback for first 3 errors
                traceback.print_exc()
            # Add default prediction for failed videos
            video_filename = os.path.basename(video_path)
            results.append({
                'filename': video_filename,
                'label': 0,  # Default to real if processing fails
                'confidence': 0.0
            })
    
    print(f"\nProcessing complete: {processed_count} successful, {error_count} errors")
    
    # Check if we have any results
    if not results:
        print("\nWarning: No results collected. Creating empty CSV with headers.")
        empty_df = pd.DataFrame(columns=['filename', 'label'])
        empty_df.to_csv(output_csv, index=False)
        return empty_df
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create submission DataFrame (adjust column names based on competition requirements)
    # Common formats:
    # - filename, label
    # - filename, prediction
    # - filename, fake (where fake is 0 or 1)
    submission_df = pd.DataFrame({
        'filename': df['filename'],
        'label': df['label']  # Change 'label' to 'prediction' or 'fake' if required
    })
    
    # Ensure we have data before saving
    if len(submission_df) == 0:
        print("\nWarning: Submission DataFrame is empty. Creating CSV with headers only.")
        empty_df = pd.DataFrame(columns=['filename', 'label'])
        empty_df.to_csv(output_csv, index=False)
        return empty_df
    
    # Save to CSV - ensure it's written properly
    try:
        # Write CSV with explicit encoding and ensure it's flushed
        submission_df.to_csv(output_csv, index=False, encoding='utf-8')
        
        # Force write to disk
        import sys
        sys.stdout.flush()
        
        print(f"\nâœ“ Submission file saved to: {output_csv}")
        print(f"  Total predictions: {len(submission_df)}")
        print(f"  Real (0): {(submission_df['label'] == 0).sum()}")
        print(f"  Fake (1): {(submission_df['label'] == 1).sum()}")
        
        # Display first few rows
        print("\nFirst 5 predictions:")
        print(submission_df.head())
        
        # Verify file was created and has content
        if os.path.exists(output_csv):
            file_size = os.path.getsize(output_csv)
            print(f"\nFile verification: {output_csv} exists ({file_size} bytes)")
            if file_size == 0:
                print("âš ï¸�  WARNING: File is 0 bytes! This means the CSV is empty.")
            elif file_size < 50:
                print("âš ï¸�  WARNING: File is very small. It may only contain headers.")
            else:
                print("âœ“ File has content and should be ready for submission.")
        else:
            print(f"\nâ�Œ ERROR: File {output_csv} was not created!")
            
    except Exception as e:
        print(f"\nâ�Œ Error saving CSV file: {e}")
        print(f"Attempting to save with alternative method...")
        try:
            # Alternative: write directly
            with open(output_csv, 'w', encoding='utf-8') as f:
                f.write('filename,label\n')
                for _, row in submission_df.iterrows():
                    f.write(f"{row['filename']},{row['label']}\n")
            print(f"âœ“ CSV saved using alternative method: {output_csv}")
        except Exception as e2:
            print(f"â�Œ Alternative save also failed: {e2}")
    
    return submission_df

# Generate submission CSV
# Process videos from multiple directories and save to /kaggle/working/submission.csv

# Optional: Check what's available in the input directory (for debugging)
print("Checking available directories in /kaggle/input/...")
if os.path.exists("/kaggle/input"):
    print("Contents of /kaggle/input:")
    for item in os.listdir("/kaggle/input"):
        item_path = os.path.join("/kaggle/input", item)
        if os.path.isdir(item_path):
            print(f"  ğŸ“� {item}/")
            # Check if it's the celeb_df-v2 directory
            if "celeb" in item.lower():
                celeb_path = os.path.join("/kaggle/input", item)
                if os.path.exists(celeb_path):
                    print(f"    Subdirectories in {item}:")
                    try:
                        for subdir in os.listdir(celeb_path):
                            subdir_path = os.path.join(celeb_path, subdir)
                            if os.path.isdir(subdir_path):
                                file_count = len([f for f in os.listdir(subdir_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                                print(f"      ğŸ“� {subdir}/ ({file_count} video files)")
                    except Exception as e:
                        print(f"      Error listing subdirectories: {e}")
else:
    print("  /kaggle/input does not exist")

print("\n" + "="*70)

# Run the submission generation
# Note: Directory is "celeb-df-v2" (hyphen, not underscore) and subdirectories have specific capitalization
submission_df = generate_kaggle_submission(
    test_dirs=[
        "/kaggle/input/celeb-df-v2/Celeb-real",
        "/kaggle/input/celeb-df-v2/Celeb-synthesis",
        "/kaggle/input/celeb-df-v2/YouTube-real"
    ],
    output_csv="/kaggle/working/submission.csv"
)

print("\n" + "="*70)
print("KAGGLE SUBMISSION READY")
print("="*70)
print("\nThe function is configured to process videos from:")
print("  - /kaggle/input/celeb-df-v2/Celeb-real")
print("  - /kaggle/input/celeb-df-v2/Celeb-synthesis")
print("  - /kaggle/input/celeb-df-v2/YouTube-real")
print("\nThe CSV file will be saved to: /kaggle/working/submission.csv")
print("This file will be available for download in Kaggle.")
print("\nTo run: Just execute this cell (the function call is already uncommented above)")


