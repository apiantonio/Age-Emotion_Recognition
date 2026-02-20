import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.config import *
from PIL import Image

# Trasformazioni
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
    ]),
    'val': transforms.Compose([
        transforms.Resize(RESIZE_SIZE),        
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets
class BaseDataset(Dataset):
    def load_image(self, path):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Errore caricamento {path}: {e}")
            return Image.new('RGB', (224, 224))

class EmotionDataset(BaseDataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = self.load_image(row['path'])
        label = torch.tensor(int(row['label_idx']), dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

class AgeDataset(BaseDataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = self.load_image(row['path'])
        age = torch.tensor(float(row['age']), dtype=torch.float32)
        weight = torch.tensor(float(row['sample_weight']), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, age, weight

# Funzioni per Caricamento Modelli e Dataloaders
def process_emotion_data(root_dir):
    """
    Gestisce il dataset Emozioni.
    - Se trova le cartelle 'train' e 'test', usa quello split (EVITA DATA LEAKAGE).
    - Altrimenti, scansiona tutto e fa uno split casuale.
    """
    print(f"Scansione Dataset Emozioni in: {root_dir}")

    # Definizione percorsi standard FER2013
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    
    # Alcune versioni chiamano il test 'validation' o 'public_test'
    if not os.path.exists(test_dir):
        if os.path.exists(os.path.join(root_dir, 'validation')):
            test_dir = os.path.join(root_dir, 'validation')

    # Helper function per leggere una cartella
    def load_from_folder(folder_path):
        paths = []
        labels = []
        # Cerca jpg, png, jpeg
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for filepath in glob.glob(os.path.join(folder_path, '**', ext), recursive=True):
                classname = os.path.basename(os.path.dirname(filepath))
                paths.append(filepath)
                labels.append(classname)
        return paths, labels

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Carica Train
        tr_paths, tr_labels = load_from_folder(train_dir)
        train_df = pd.DataFrame({'path': tr_paths, 'label': tr_labels})
        
        # Carica Val (Test)
        val_paths, val_labels = load_from_folder(test_dir)
        val_df = pd.DataFrame({'path': val_paths, 'label': val_labels})

        # Uniamo le labels per essere sicuri di avere tutte le classi (mappatura coerente)
        all_labels = sorted(list(set(tr_labels + val_labels)))
        class_to_idx = {cls: i for i, cls in enumerate(all_labels)}
            
    # Applica mappatura stringa -> int
    train_df['label_idx'] = train_df['label'].map(class_to_idx)
    val_df['label_idx'] = val_df['label'].map(class_to_idx)

    # Controllo integrità (rimuove eventuali NaN se ci sono classi nel test non presenti nel train)
    train_df = train_df.dropna(subset=['label_idx'])
    val_df = val_df.dropna(subset=['label_idx'])
    
    # Converti in int
    train_df['label_idx'] = train_df['label_idx'].astype(int)
    val_df['label_idx'] = val_df['label_idx'].astype(int)

    # Calcolo Pesi Classi (SOLO SUL TRAIN)
    train_labels = train_df['label_idx'].values
    unique_classes = np.unique(train_labels)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels
    )

    # Se mancano classi nel train rispetto alla mappa totale, dobbiamo gestire i pesi
    # Creiamo un array di pesi della dimensione giusta (len(class_to_idx))
    final_weights = np.ones(len(class_to_idx), dtype=np.float32)
    for cls_idx, weight in zip(unique_classes, class_weights):
        final_weights[cls_idx] = weight

    weights_tensor = torch.tensor(final_weights, dtype=torch.float32)

    print(f"   Classi trovate: {len(class_to_idx)}")
    print(f"   Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(f"   Pesi calcolati: {weights_tensor}")

    return train_df, val_df, weights_tensor, class_to_idx

# Funzione per Caricare Modelli e Trasformazioni
def process_age_data(root_dir):
    print(f"Scansione Dataset Età in: {root_dir}")
    
    if not os.path.exists(root_dir):
         raise FileNotFoundError(f"Directory {root_dir} non trovata.")
         
    files = glob.glob(os.path.join(root_dir, "*.jpg"))
    if not files:
        raise ValueError("Nessuna immagine trovata nel percorso specificato.")
        
    ages = []; paths = []
    for f in files:
        try:
            age = int(os.path.basename(f).split('_')[0])
            ages.append(age)
            paths.append(f)
        except: pass

    df = pd.DataFrame({'path': paths, 'age': ages})
    bins = list(range(0, 81, 5)) + [120]
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=False, include_lowest=True)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['age_bin'], random_state=RANDOM_SEED)

    bin_counts = train_df['age_bin'].value_counts().sort_index()
    weights_per_bin = 1.0 / bin_counts
    weights_per_bin = weights_per_bin / weights_per_bin.mean()

    train_df['sample_weight'] = train_df['age_bin'].map(weights_per_bin)
    val_df['sample_weight'] = 1.0

    return train_df, val_df


## Entry point per i dataloader
def get_dataloaders(task, batch_size=BATCH_SIZE):
    """
    Funzione principale da chiamare per ottenere i dataloader
    """
    dl_kwargs = {
        'num_workers': 4,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': True,
        'prefetch_factor': None # 1 if torch.cuda.is_available() else None
    }

    if task == 'emotion':
        # Gestione Emozioni
        if os.path.exists(EMO_TRAIN_CSV) and os.path.exists(EMO_META_PT):
            print("Caricamento split Emozioni esistenti...")
            train_df = pd.read_csv(EMO_TRAIN_CSV)
            val_df = pd.read_csv(EMO_VAL_CSV)
            meta = torch.load(EMO_META_PT, weights_only=False)
            weights = meta['weights']
        else:
            print("Creazione nuovi split Emozioni...")
            train_df, val_df, weights, class_map = process_emotion_data(FER_ROOT_PATH)
            train_df.to_csv(EMO_TRAIN_CSV, index=False)
            val_df.to_csv(EMO_VAL_CSV, index=False)
            torch.save({'weights': weights, 'class_map': class_map}, EMO_META_PT)

        train_ds = EmotionDataset(train_df, transform=data_transforms['train'])
        val_ds = EmotionDataset(val_df, transform=data_transforms['val'])
        
        # CrossEntropy vuole i pesi, ritorniamo weights
        class_weights_out = weights 

    elif task == 'age':
        # Gestione Età
        if os.path.exists(AGE_TRAIN_CSV):
            print("Caricamento split Età esistenti...")
            train_df = pd.read_csv(AGE_TRAIN_CSV)
            val_df = pd.read_csv(AGE_VAL_CSV)
        else:
            print("Creazione nuovi split Età...")
            train_df, val_df = process_age_data(UTKFACE_PATH)
            train_df.to_csv(AGE_TRAIN_CSV, index=False)
            val_df.to_csv(AGE_VAL_CSV, index=False)

        train_ds = AgeDataset(train_df, transform=data_transforms['train'])
        val_ds = AgeDataset(val_df, transform=data_transforms['val'])
        
        # La regressione usa sample_weights internamente, non class_weights
        class_weights_out = None 

    else:
        raise ValueError(f"Task {task} non valido")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)

    print(f"Dataloaders pronti per {task.upper()}")
    print(f"   Train: {len(train_ds)} sample | Val: {len(val_ds)} sample")
    
    return train_loader, val_loader, class_weights_out