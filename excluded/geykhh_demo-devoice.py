# BirdCLEF_Demucs_Demo.ipynb

# 安装必要的库
!pip install -q demucs librosa matplotlib soundfile torchaudio

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio
import soundfile as sf
from IPython.display import Audio, display
from pathlib import Path

# 确保我们使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载Demucs模型
print("加载Demucs模型...")
from demucs.pretrained import get_model
from demucs.apply import apply_model

# 使用htdemucs模型，它在分离人声和乐器方面表现非常好
model = get_model("htdemucs")
model.to(device)
print(f"模型加载完成: {model.sources}")  # 通常包括['drums', 'bass', 'other', 'vocals']




# 修改visualize_audio函数，使用英文显示
def visualize_audio(audio_data, sr, title="Audio Waveform and Spectrogram"):
    """
    Display audio waveform and spectrogram
    显示音频波形和频谱图
    """
    plt.figure(figsize=(14, 6))
    
    # Waveform (波形图)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title(f"{title} - Waveform")  # 波形图标题
    plt.xlabel("Time (s)")  # 时间轴标签
    plt.ylabel("Amplitude")  # 振幅轴标签
    
    # Spectrogram (频谱图)
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title} - Spectrogram")  # 频谱图标题
    plt.tight_layout()
    plt.show()

# 修改separate_voice函数，使用英文显示
def separate_voice(audio_path, output_dir="separated_audio"):
    """
    Use Demucs to separate vocals from other sounds in audio
    使用Demucs分离音频中的人声和其他声音
    
    Parameters (参数):
    audio_path: Path to audio file (音频文件路径)
    output_dir: Directory to save separated audio (分离后的音频保存目录)
    
    Returns (返回):
    Dictionary of separated audio sources (分离后的音频字典)
    Sample rate (采样率)
    """
    # Create output directory (创建输出目录)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio (加载音频)
    print(f"Loading audio file: {audio_path}")  # 加载音频文件
    try:
        # Use librosa to load audio, keeping original sample rate (使用librosa加载音频，保持原采样率)
        audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Print audio information (打印音频信息)
        print(f"Audio info: Sample rate={sr}Hz, Duration={len(audio_data)/sr:.2f}s")
        
        # Visualize original audio (可视化原始音频)
        visualize_audio(audio_data, sr, title="Original Audio")  # 原始音频
        display(Audio(audio_data, rate=sr))
        
        # Fix: Convert mono to stereo by duplicating the channel (通过复制通道将单声道转换为立体声)
        # Demucs expects input shape [batch, channels, time] with channels=2
        stereo_audio = np.stack([audio_data, audio_data])  # 复制单声道到两个通道
        audio_tensor = torch.tensor(stereo_audio, device=device).unsqueeze(0)  # 形状变为 [1, 2, time]
        
        print(f"Converted audio shape: {audio_tensor.shape}")  # 转换后的音频形状
        
        # Separate audio (分离音频)
        print("Separating audio with Demucs...")  # 使用Demucs分离音频
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device=device)[0]
        
        # Extract each part from sources (从sources中提取各个部分)
        # 4 sources: drums, bass, other, vocals
        sources_dict = {}
        for source, source_audio in zip(model.sources, sources):
            # Take first channel of stereo output (取立体声输出的第一个通道)
            source_audio_mono = source_audio[0].cpu().numpy()
            sources_dict[source] = source_audio_mono
            
            # Save separated audio (保存分离的音频)
            output_path = os.path.join(output_dir, f"{Path(audio_path).stem}_{source}.wav")
            sf.write(output_path, source_audio_mono, sr)
            print(f"Saved separated audio: {output_path}")  # 已保存分离的音频
        
        # Create no-vocals version (combining all sources except vocals)
        # 创建无人声版本 (合并除vocals外的所有源)
        no_vocals = sum([sources_dict[source] for source in model.sources if source != 'vocals'])
        sources_dict['no_vocals'] = no_vocals
        
        # Save no-vocals version (保存无人声版本)
        output_path = os.path.join(output_dir, f"{Path(audio_path).stem}_no_vocals.wav")
        sf.write(output_path, no_vocals, sr)
        print(f"Saved no-vocals audio: {output_path}")  # 已保存无人声音频
        
        # Visualize separated audio (可视化分离后的音频)
        vocals = sources_dict['vocals']
        visualize_audio(vocals, sr, title="Separated Vocals")  # 分离出的人声
        display(Audio(vocals, rate=sr))
        
        visualize_audio(no_vocals, sr, title="Audio without Vocals")  # 无人声音频
        display(Audio(no_vocals, rate=sr))
        
        return sources_dict, sr
        
    except Exception as e:
        print(f"Error processing audio: {e}")  # 处理音频时出错
        import traceback
        traceback.print_exc()
        return None, None

# 修改compare_spectrograms函数，使用英文显示
def compare_spectrograms(audio_original, audio_no_vocals, sr, title="Original vs No-Vocals Audio"):
    """
    Compare mel spectrograms of two audio samples
    比较两个音频的梅尔频谱图
    """
    # Calculate mel spectrograms (计算梅尔频谱图)
    mel_spec_orig = librosa.feature.melspectrogram(
        y=audio_original, sr=sr, n_fft=1024, hop_length=512, 
        n_mels=128, fmin=50, fmax=14000
    )
    mel_spec_no_vocals = librosa.feature.melspectrogram(
        y=audio_no_vocals, sr=sr, n_fft=1024, hop_length=512, 
        n_mels=128, fmin=50, fmax=14000
    )
    
    # Convert to decibels (转换为分贝)
    mel_spec_db_orig = librosa.power_to_db(mel_spec_orig, ref=np.max)
    mel_spec_db_no_vocals = librosa.power_to_db(mel_spec_no_vocals, ref=np.max)
    
    # Visualize comparison (可视化比较)
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        mel_spec_db_orig, sr=sr, hop_length=512, 
        x_axis='time', y_axis='mel', fmin=50, fmax=14000
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Audio Mel Spectrogram')  # 原始音频的梅尔频谱图
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        mel_spec_db_no_vocals, sr=sr, hop_length=512, 
        x_axis='time', y_axis='mel', fmin=50, fmax=14000
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('No-Vocals Audio Mel Spectrogram')  # 无人声音频的梅尔频谱图
    
    plt.tight_layout()
    plt.show()
    
    # Calculate spectrogram difference (vocals portion)
    # 计算频谱图差异（人声部分）
    diff = mel_spec_db_orig - mel_spec_db_no_vocals
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        diff, sr=sr, hop_length=512, 
        x_axis='time', y_axis='mel', fmin=50, fmax=14000
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Difference (Original - No Vocals)')  # 频谱图差异 (原始 - 无人声)
    plt.tight_layout()
    plt.show()
    
    return mel_spec_db_orig, mel_spec_db_no_vocals, diff

# 修改主测试函数
def test_demucs_on_samples(sample_paths):
    """
    Test Demucs separation on multiple samples
    在多个样本上测试Demucs分离效果
    """
    results = []
    
    for audio_path in sample_paths:
        print(f"\nProcessing sample: {audio_path}")  # 处理样本
        try:
            # Separate vocals (分离人声)
            sources_dict, sr = separate_voice(audio_path)
            
            if sources_dict and sr:
                # Compare spectrograms (比较频谱图)
                original_audio = librosa.load(audio_path, sr=sr)[0]
                no_vocals_audio = sources_dict['no_vocals']
                
                mel_orig, mel_no_vocals, diff = compare_spectrograms(original_audio, no_vocals_audio, sr)
                
                results.append({
                    'path': audio_path,
                    'sources': sources_dict,
                    'sr': sr,
                    'mel_original': mel_orig,
                    'mel_no_vocals': mel_no_vocals,
                    'mel_diff': diff
                })
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")  # 处理时出错
    
    return results

# 主程序更新
sample_paths = [
    # Replace with actual file paths (替换为实际的文件路径)
    # Recommended to use Fabio A. Sarria-S recordings (推荐选择Fabio A. Sarria-S的录音)
    '/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg',
    '/kaggle/input/birdclef-2025/train_audio/1139490/CSA36389.ogg',
]

# Run test (运行测试)
print("Testing Demucs voice separation...")  # 测试Demucs人声分离
results = test_demucs_on_samples(sample_paths[:2])  # 只测试前两个样本以节省时间

print("\nSeparation complete! Check the output directory for the separated audio files.")
# 分离完成! 查看输出目录中的文件以获取分离后的音频。


# 从人声数据文件中加载样本
print("\n从已知含人声的音频文件中测试Demucs")

import pickle
import random
# 修正提取相对路径的方法
def extract_relative_path(full_path):
    """从完整路径中提取出train_audio/之后的相对路径"""
    if 'train_audio/' in full_path:
        return full_path.split('train_audio/')[-1]
    return full_path
# 加载已知包含人声的文件列表
try:
    with open(voice_files_path, 'r') as f:
        voice_files = [line.strip() for line in f.readlines()]
    print(f"成功加载了 {len(voice_files)} 个已知包含人声的文件")
    
    # 修正路径 - 提取train_audio/后面的部分作为相对路径
    voice_files = [extract_relative_path(path) for path in voice_files]
    
    # 加载人声时间戳数据（可选，用于更详细的分析）
    try:
        with open(voice_data_path, 'rb') as f:
            voice_timestamps = pickle.load(f)
        print("成功加载人声时间戳数据")
        has_timestamps = True
    except Exception as e:
        print(f"加载人声时间戳数据时出错: {e}")
        voice_timestamps = {}
        has_timestamps = False
    
    # 加载训练数据
    train_csv_path = '/kaggle/input/birdclef-2025/train.csv'
    train_df = pd.read_csv(train_csv_path)
    
    # 为每个评分级别随机选择含人声的样本
    random.seed(42)  # 固定随机种子
    
    # 找出文件名在voice_files中的样本
    voice_files_set = set(voice_files)
    voice_samples_df = train_df[train_df['filename'].apply(lambda x: x in voice_files_set)]
    print(f"找到 {len(voice_samples_df)} 个匹配的样本")
    
    # 按评分分组，并为每个评分选择样本
    rating_samples = {}
    for rating in range(6):  # 0 to 5
        rating_df = voice_samples_df[voice_samples_df['rating'] == rating]
        
        if len(rating_df) > 0:
            # 如果评分组有多个样本，随机选择最多3个
            if len(rating_df) > 3:
                samples = rating_df.sample(3, random_state=42)
            else:
                samples = rating_df
                
            paths = [f'/kaggle/input/birdclef-2025/train_audio/{filename}' for filename in samples['filename']]
            rating_samples[rating] = paths
            print(f"评分 {rating}: 选择了 {len(paths)} 个含人声的样本")
        else:
            rating_samples[rating] = []
            print(f"评分 {rating}: 未找到含人声的样本")
    
    # 如果某些评分级别没有含人声的样本，从所有含人声样本中随机选择
    if any(len(paths) == 0 for rating, paths in rating_samples.items()):
        # 随机打乱样本
        all_voice_paths = [f'/kaggle/input/birdclef-2025/train_audio/{filename}' for filename in voice_samples_df['filename']]
        random.shuffle(all_voice_paths)
        
        # 为空的评分组分配样本
        for rating in rating_samples:
            if len(rating_samples[rating]) == 0 and all_voice_paths:
                rating_samples[rating] = [all_voice_paths.pop()]
                print(f"评分 {rating}: 从所有含人声样本中随机分配了1个样本")
    
    # 按评分测试样本
    print("\n开始按评分级别测试含人声的样本...")
    for rating, paths in rating_samples.items():
        if not paths:
            print(f"\n跳过评分 {rating}: 没有可用样本")
            continue
            
        print(f"\n### 测试评分为 {rating} 的含人声样本 ###")
        
        # 为每个评分仅测试一个样本以节省时间
        sample_path = paths[0]
        try:
            print(f"处理样本: {sample_path}")
            
            # 如果有时间戳数据，显示人声出现的时间段
            if has_timestamps and sample_path in voice_timestamps:
                timestamps = voice_timestamps[sample_path]
                print(f"人声出现在以下时间段（秒）:")
                for ts in timestamps[:5]:  # 最多显示5个时间戳
                    print(f"  {ts['start']:.2f} - {ts['end']:.2f}")
                if len(timestamps) > 5:
                    print(f"  ... 共 {len(timestamps)} 个人声片段")
            
            # 分离音频
            sources_dict, sr = separate_voice(sample_path)
            
            if sources_dict and sr:
                # 比较频谱图
                original_audio = librosa.load(sample_path, sr=sr)[0]
                no_vocals_audio = sources_dict['no_vocals']
                
                # 计算梅尔频谱图
                mel_orig, mel_no_vocals, diff = compare_spectrograms(original_audio, no_vocals_audio, sr)
                
                # 显示分离的听觉效果
                print("原始音频:")
                display(Audio(original_audio, rate=sr))
                
                print("分离出的人声:")
                display(Audio(sources_dict['vocals'], rate=sr))
                
                print("无人声音频（鸟叫声）:")
                display(Audio(no_vocals_audio, rate=sr))
                
        except Exception as e:
            print(f"处理评分为 {rating} 的样本时出错: {e}")
            import traceback
            traceback.print_exc()

    # 如果要测试具体的样本（不按评分分组）
    print("\n测试几个特定的人声样本...")
    
    # 随机从可用样本中选择不超过5个
    test_samples = random.sample(all_voice_paths, min(5, len(all_voice_paths)))
    
    for i, sample_path in enumerate(test_samples):
        print(f"\n### 测试样本 {i+1}/{len(test_samples)} ###")
        try:
            print(f"处理样本: {sample_path}")
            
            # 分离音频
            sources_dict, sr = separate_voice(sample_path)
            
            if sources_dict and sr:
                # 比较频谱图
                original_audio = librosa.load(sample_path, sr=sr)[0]
                no_vocals_audio = sources_dict['no_vocals']
                vocals_audio = sources_dict['vocals']
                
                # 计算梅尔频谱图
                mel_orig, mel_no_vocals, diff = compare_spectrograms(original_audio, no_vocals_audio, sr)
                
                # 显示分离的听觉效果
                print("原始音频:")
                display(Audio(original_audio, rate=sr))
                
                print("分离出的人声:")
                display(Audio(vocals_audio, rate=sr))
                
                print("无人声音频（鸟叫声）:")
                display(Audio(no_vocals_audio, rate=sr))
        except Exception as e:
            print(f"处理样本时出错: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"处理人声数据文件时出错: {e}")
    import traceback
    traceback.print_exc()

print("\n基于人声数据的测试完成!")


# 在实际项目中集成Demucs
print("\n如何在你的BirdCLEF项目中集成Demucs:")
print("""
# 1. 创建一个过滤人声的函数
def filter_vocals_with_demucs(audio_path, sr=32000):
    # 加载音频
    audio_tensor, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_sr, sr)
    
    # 确保音频是单声道
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    
    # 分离音频
    with torch.no_grad():
        sources = apply_model(model, audio_tensor.to(device), device=device)[0]
    
    # 创建无人声版本 (合并除vocals外的所有源)
    sources_dict = {}
    for i, source in enumerate(model.sources):
        sources_dict[source] = sources[i].cpu()
    
    # 合并非人声部分
    no_vocals = sum([sources_dict[source] for source in model.sources if source != 'vocals'])
    
    return no_vocals.numpy()

# 2. 在Dataset类中使用这个函数
# 在BirdCLEFDataset类的__getitem__方法中:

def __getitem__(self, idx):
    # ... 现有代码 ...
    
    # 如果需要处理人声
    if self.remove_vocals and os.path.exists(row['filepath']):
        # 使用Demucs去除人声
        audio_no_vocals = filter_vocals_with_demucs(row['filepath'])
        
        # 使用无人声音频生成频谱图
        spec = self._audio_to_melspec(audio_no_vocals)
    else:
        # 使用原始方法处理音频
        spec = process_audio_file(row['filepath'], self.cfg)
    
    # ... 剩余代码 ...
""")

