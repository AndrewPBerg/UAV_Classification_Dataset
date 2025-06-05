# Purpose of this script is to generate all of the feature extracted spectrograms from the audio files. 
# Note that the audio files are not included in this repository

# Reference https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html for details on the transforms

import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram, LFCC
import torchaudio.functional as F
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import sys
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract and save audio features (LFCC, Mel Spectrogram, MFCC, Spectrogram, Mel Filter Bank) as images."""
    
    # Feature type constants
    FEATURE_TYPES = ['LFCC', 'MEL_SPECTROGRAM', 'MFCC', 'SPECTROGRAM', 'MEL_FILTER_BANK']
    
    # Progress emojis
    EMOJIS = {
        'start': '🚀',
        'processing': '⚙️',
        'success': '✅',
        'skip': '⏭️',
        'error': '❌',
        'complete': '🎉',
        'folder': '📁',
        'audio': '🎵',
        'image': '🖼️',
        'validation': '🔍'
    }
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.current_dir = Path(__file__).resolve().parent
        # Point to the external dataset directory
        self.datasets_dir = Path(r'C:\Users\Sidewinders\Desktop\CODE\UAV_Classification_repo\src\datasets') / dataset_name
        self.output_base_dir = self.current_dir  # Put feature dirs directly in spectrogram_dataset
        
        # Audio processing parameters
        self.sampling_rate = 16000
        self.n_mels = 128  # Keep for mel spectrogram
        self.n_mels_mfcc = 256  # For MFCC as per documentation
        self.n_fft = 2048  # Updated to match documentation
        self.hop_length = 512
        self.power = 2.0
        self.n_mfcc = 256  # Updated to match documentation
        self.n_lfcc = 256  # Updated to match documentation
        
        # Initialize transforms
        self._init_transforms()
        
        # Track processed classes
        self.existing_classes = self._get_existing_classes()
        
    def _init_transforms(self):
        """Initialize audio feature transforms using torchaudio abstractions."""
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power
        )
        
        self.mfcc_transform = MFCC(
            sample_rate=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels_mfcc
            }
        )
        
        self.spectrogram_transform = Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=self.power
        )
        
        self.lfcc_transform = LFCC(
            sample_rate=self.sampling_rate,
            n_lfcc=self.n_lfcc,
            speckwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
            }
        )
        
        # Create mel filter bank using torchaudio functional
        self.mel_filter_bank = F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0.0,
            f_max=self.sampling_rate / 2.0,
            n_mels=self.n_mels,
            sample_rate=self.sampling_rate,
            norm="slaney"
        )
    
    def _get_existing_classes(self) -> Dict[str, Set[str]]:
        """Get already processed classes for each feature type."""
        existing = {feature_type: set() for feature_type in self.FEATURE_TYPES}
        
        if self.output_base_dir.exists():
            for feature_type in self.FEATURE_TYPES:
                feature_dir = self.output_base_dir / feature_type
                if feature_dir.exists():
                    for class_dir in feature_dir.iterdir():
                        if class_dir.is_dir() and list(class_dir.glob("*.png")):
                            existing[feature_type].add(class_dir.name)
        
        return existing
    
    def _compute_mel_filter_bank(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel filter bank features using torchaudio functional."""
        # For MEL_FILTER_BANK, we return the filter bank matrix itself, not applied to audio
        # This matches the PyTorch documentation approach
        return self.mel_filter_bank.T.unsqueeze(0)  # Add batch dimension for consistency
    
    def get_audio_files(self, root_dir: Path) -> List[Path]:
        """Get all audio files recursively."""
        return list(root_dir.rglob("*.wav"))
    
    def _should_process_class(self, class_name: str, feature_type: str) -> bool:
        """Check if a class should be processed for a given feature type."""
        return class_name not in self.existing_classes[feature_type]
    
    def _create_output_dirs(self, class_name: str) -> Dict[str, Path]:
        """Create output directories for all feature types."""
        output_dirs = {}
        for feature_type in self.FEATURE_TYPES:
            feature_dir = self.output_base_dir / feature_type / class_name
            feature_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[feature_type] = feature_dir
        return output_dirs
    
    def _save_feature_plot(self, feature_data: torch.Tensor, feature_type: str, 
                          audio_path: Path, output_dir: Path) -> Path:
        """Save feature data as a plot image."""
        plt.figure(figsize=(12, 8))
        
        if feature_type == 'MEL_SPECTROGRAM':
            plt.imshow(librosa.power_to_db(feature_data[0].numpy()), origin="lower", aspect="auto", interpolation="nearest")
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram - {audio_path.stem}')
            plt.ylabel('mel freq')
            plt.xlabel('Time Frames')
        elif feature_type == 'MFCC':
            plt.imshow(librosa.power_to_db(feature_data[0].numpy()), origin="lower", aspect="auto", interpolation="nearest")
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'MFCC - {audio_path.stem}')
            plt.ylabel('mfcc')
            plt.xlabel('Time Frames')
        elif feature_type == 'LFCC':
            plt.imshow(librosa.power_to_db(feature_data[0].numpy()), origin="lower", aspect="auto", interpolation="nearest")
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'LFCC - {audio_path.stem}')
            plt.ylabel('lfcc')
            plt.xlabel('Time Frames')
        elif feature_type == 'SPECTROGRAM':
            plt.imshow(librosa.power_to_db(feature_data[0].numpy()), origin="lower", aspect="auto", interpolation="nearest")
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram - {audio_path.stem}')
            plt.ylabel('freq_bin')
            plt.xlabel('Time Frames')
        elif feature_type == 'MEL_FILTER_BANK':
            # Plot the mel filter bank matrix following PyTorch documentation approach
            plt.imshow(feature_data[0].numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Mel Filter Bank - {audio_path.stem}')
            plt.ylabel('frequency bin')
            plt.xlabel('mel bin')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / f"{audio_path.stem}_{feature_type.lower()}.png"
        plt.savefig(str(output_path), format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def process_audio_file(self, audio_path: Path, output_dirs: Dict[str, Path]) -> Dict[str, bool]:
        """Process a single audio file and generate all feature types."""
        results = {}
        
        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(str(audio_path))
            logger.info(f"    {self.EMOJIS['audio']} Loaded: {audio_path.name}")
            
            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info(f"    {self.EMOJIS['processing']} Converted to mono from {waveform.shape[0]} channels")
            
            # Generate features using torchaudio transforms
            features = {}
            features['MEL_SPECTROGRAM'] = self.mel_transform(waveform)
            features['MFCC'] = self.mfcc_transform(waveform)
            features['LFCC'] = self.lfcc_transform(waveform)
            features['SPECTROGRAM'] = self.spectrogram_transform(waveform)
            features['MEL_FILTER_BANK'] = self._compute_mel_filter_bank(waveform)
            
            # Save each feature type
            for feature_type, feature_data in features.items():
                try:
                    output_path = self._save_feature_plot(
                        feature_data, feature_type, audio_path, output_dirs[feature_type]
                    )
                    results[feature_type] = True
                    logger.info(f"      {self.EMOJIS['image']} {feature_type}: {output_path.name}")
                except Exception as e:
                    logger.error(f"      {self.EMOJIS['error']} Failed {feature_type}: {str(e)}")
                    results[feature_type] = False
            
        except Exception as e:
            logger.error(f"    {self.EMOJIS['error']} Error processing {audio_path}: {str(e)}")
            for feature_type in self.FEATURE_TYPES:
                results[feature_type] = False
        
        return results
    
    def run(self):
        """Main execution method."""
        start_time = datetime.now()
        logger.info(f"{self.EMOJIS['start']} Starting feature extraction for dataset: {self.dataset_name}")
        logger.info(f"{self.EMOJIS['folder']} Dataset directory: {self.datasets_dir}")
        logger.info(f"{self.EMOJIS['folder']} Output directory: {self.output_base_dir}")
        
        # Validate dataset directory
        if not self.datasets_dir.exists():
            logger.error(f"{self.EMOJIS['error']} Dataset directory not found: {self.datasets_dir}")
            return
        
        # Get all audio files
        audio_files = self.get_audio_files(self.datasets_dir)
        logger.info(f"{self.EMOJIS['audio']} Found {len(audio_files)} audio files")
        
        # Group files by class (parent directory)
        classes = {}
        for audio_path in audio_files:
            class_name = audio_path.parent.name
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append(audio_path)
        
        logger.info(f"{self.EMOJIS['folder']} Found {len(classes)} classes: {list(classes.keys())}")
        
        # Show existing classes validation
        logger.info(f"{self.EMOJIS['validation']} Validating existing processed classes...")
        for feature_type in self.FEATURE_TYPES:
            existing = self.existing_classes[feature_type]
            if existing:
                logger.info(f"  {feature_type}: {len(existing)} classes already processed {list(existing)}")
            else:
                logger.info(f"  {feature_type}: No existing processed classes")
        
        # Process each class
        total_stats = {feature_type: {'processed': 0, 'skipped': 0, 'errors': 0} 
                      for feature_type in self.FEATURE_TYPES}
        
        for class_idx, (class_name, audio_files_in_class) in enumerate(classes.items(), 1):
            logger.info(f"\n{self.EMOJIS['processing']} Processing class {class_idx}/{len(classes)}: {class_name}")
            logger.info(f"  Files in class: {len(audio_files_in_class)}")
            
            # Check which feature types need processing for this class
            needs_processing = {}
            for feature_type in self.FEATURE_TYPES:
                needs_processing[feature_type] = self._should_process_class(class_name, feature_type)
                if not needs_processing[feature_type]:
                    logger.info(f"  {self.EMOJIS['skip']} {feature_type}: Already processed, skipping")
                    total_stats[feature_type]['skipped'] += len(audio_files_in_class)
            
            # Skip if all feature types are already processed
            if not any(needs_processing.values()):
                logger.info(f"  {self.EMOJIS['skip']} All feature types already processed for {class_name}")
                continue
            
            # Create output directories
            output_dirs = self._create_output_dirs(class_name)
            
            # Process each audio file in the class
            for file_idx, audio_path in enumerate(audio_files_in_class, 1):
                logger.info(f"  {self.EMOJIS['processing']} File {file_idx}/{len(audio_files_in_class)}: {audio_path.name}")
                
                # Process only needed feature types
                file_results = self.process_audio_file(audio_path, output_dirs)
                
                # Update statistics
                for feature_type in self.FEATURE_TYPES:
                    if needs_processing[feature_type]:
                        if file_results.get(feature_type, False):
                            total_stats[feature_type]['processed'] += 1
                        else:
                            total_stats[feature_type]['errors'] += 1
            
            # Mark class as processed for successfully completed feature types
            for feature_type in self.FEATURE_TYPES:
                if needs_processing[feature_type]:
                    self.existing_classes[feature_type].add(class_name)
            
            logger.info(f"  {self.EMOJIS['success']} Completed class: {class_name}")
        
        # Final statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n{self.EMOJIS['complete']} Feature extraction completed!")
        logger.info(f"Total execution time: {duration}")
        logger.info("\nProcessing Statistics:")
        
        for feature_type in self.FEATURE_TYPES:
            stats = total_stats[feature_type]
            total_files = stats['processed'] + stats['skipped'] + stats['errors']
            logger.info(f"  {feature_type}:")
            logger.info(f"    {self.EMOJIS['success']} Processed: {stats['processed']}")
            logger.info(f"    {self.EMOJIS['skip']} Skipped: {stats['skipped']}")
            logger.info(f"    {self.EMOJIS['error']} Errors: {stats['errors']}")
            logger.info(f"    Total: {total_files}")


def main():
    """Main function to run feature extraction."""
    # You can modify this to change which dataset to process
    # DATASET_NAME = 'UAV_Dataset_31'  # Change to 'UAV_Dataset_9' or other dataset names
    DATASET_NAME = 'DJI_Neo'  # Change to 'UAV_Dataset_9' or other dataset names
    
    logger.info(f"🎯 Target dataset: {DATASET_NAME}")
    
    try:
        extractor = FeatureExtractor(DATASET_NAME)
        extractor.run()
    except KeyboardInterrupt:
        logger.info(f"\n⏹️ Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise


if __name__ == '__main__':
    main() 