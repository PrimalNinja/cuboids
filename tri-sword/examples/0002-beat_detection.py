%%writefile beat_detection.py
import torch

def beat_detection(data):
    """
    Baseline: Detect kick drums in audio spectrogram
    Input: Batch of spectrograms (batch_size x (freq_bins * time_slices))
    Output: Number of kick drums detected
    """
    batch_size = data.size(0)
    total_features = data.size(1)
    
    freq_bins = 128
    time_slices = total_features // freq_bins
    
    spec = data.view(batch_size, time_slices, freq_bins)
    
    bass_energy = spec[:, :, :20].sum(dim=2).sum(dim=1)
    high_energy = spec[:, :, 60:].sum(dim=2).sum(dim=1)
    
    kicks_detected = ((bass_energy > 40.0) & (high_energy < 20.0)).sum()
    
    return kicks_detected