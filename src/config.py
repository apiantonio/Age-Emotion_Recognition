import os
import torch
from pathlib import Path
from datetime import datetime

UTKFACE_PATH = './data/utkface/UTKFace'  # Assicurati che il path sia corretto
FER_ROOT_PATH = './data/fer2013'         # Root che contiene 'train' e 'test' (o sottocartelle classi)
BATCH_SIZE = 128
RANDOM_SEED = 42

# Percorsi per il salvataggio dei metadata (per riproducibilità)
METADATA_DIR = './metadata_splits'
os.makedirs(METADATA_DIR, exist_ok=True)

# Nomi file per il salvataggio
EMO_TRAIN_CSV = os.path.join(METADATA_DIR, 'train_emotion.csv')
EMO_VAL_CSV   = os.path.join(METADATA_DIR, 'val_emotion.csv')
EMO_META_PT   = os.path.join(METADATA_DIR, 'emotion_meta.pt') # Per salvare pesi e mappa classi

AGE_TRAIN_CSV = os.path.join(METADATA_DIR, 'train_age.csv')
AGE_VAL_CSV   = os.path.join(METADATA_DIR, 'val_age.csv')

RESIZE_SIZE = 384
CROP_SIZE = 384

# Configurazione modelli e checkpoint
MODELS_CONFIG = {
    'emotion': {
        'model_name': 'convnext_tiny',
        'checkpoint': 'checkpoints/emotion_best/best_model.pth', 
        'num_classes': 7,
        'resize': 236,
        'crop': 224,
        'labels': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    },
    'age': {
        'model_name': 'efficientnet_v2_s',
        'checkpoint': 'checkpoints/age_best/best_model.pth',   
        'num_classes': 1,
        'resize': 384,
        'crop': 384,
        'labels': None 
    }
}


class TrainingConfig:
    def __init__(self, task='emotion', resume=False):
        self.task = task
        self.num_epochs = 50    
        self.batch_size = BATCH_SIZE
        self.learning_rate = 1e-4
        self.weight_decay = 5e-2
        self.patience = 10
        self.min_delta = 0.001
        self.scheduler = 'plateau' # 'onecycle'
        self.label_smoothing = 0.1
        self.resume = resume

        # Strategia Unfreezing
        # freeze_epochs indica per quante epoche la backbone rimane COMPLETAMENTE congelata.
        # Consigliato: 1 (Epoca 0 frozen, dall'Epoca 1 inizia a scongelare gradualmente)
        self.freeze_epochs = 1

        # Directory per i salvataggi
        # Se resume=True, dovrai impostare manualmente la cartella del run precedente
        base_dir = Path('checkpoints')
        self.save_dir = base_dir / f'{task}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()

        print(f"{'='*60}")
        print(f"TRAINING CONFIGURATION - {self.task.upper()}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs} | Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate} | Weight Decay: {self.weight_decay}")
        print(f"Early Stopping Patience: {self.patience}")
        print(f"Freeze Strategy: First {self.freeze_epochs} epochs frozen")
        print(f"Save Directory: {self.save_dir}")
        print(f"AMP Enabled: {self.use_amp}")
        print(f"{'='*60}\n")