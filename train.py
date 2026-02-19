import argparse
import sys
import os

# Aggiungiamo la cartella corrente al path per permettere import corretti
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import TrainingConfig
from src.dataset import get_dataloaders
from src.model import FaceModel
from src.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Training Script per Emotion/Age Recognition")
    
    # Argomenti Obbligatori
    parser.add_argument('--task', type=str, required=True, choices=['emotion', 'age'], 
                        help="Il task su cui fare training: 'emotion' o 'age'")
    
    # Argomenti Opzionali (Hyperparameters)
    parser.add_argument('--epochs', type=int, help="Numero totale di epoche", required=False)
    parser.add_argument('--batch_size', type=int, default=64, help="Dimensione del batch")
    parser.add_argument('--model', type=str, default='resnet18', help="Architettura del modello")
    parser.add_argument('--resume', type=str, default=None, 
                        help="Path di un checkpoint .pth da cui riprendere il training", required=False)
    parser.add_argument('--patience', type=int, help="Numero di epoche di pazienza per l'early stopping", required=False)
    parser.add_argument('--freeze_epochs', type=int, default=1, 
                        help="Numero di epoche iniziali con backbone completamente congelato (default: 1)", required=False)
    # In src/train.py, dentro main():
    parser.add_argument('--scheduler', type=str, default='onecycle', 
                        choices=['onecycle', 'plateau'], 
                        help="Tipo di scheduler: 'onecycle' o 'plateau'")
    
    
    args = parser.parse_args()
    
    print(f"Avvio Training Script per task: {args.task.upper()}")
    
    # 1. Preparazione Dati
    try:
        train_loader, val_loader, class_weights = get_dataloaders(args.task, batch_size=args.batch_size)
    except Exception as e:
        print(f"Errore nella creazione dei Dataloader: {e}")
        return

    # 2. Configurazione Modello
    if args.task == 'emotion':
        num_classes = 7
        mode = 'classification'
    else:
        num_classes = 1 # Dummy per regressione
        mode = 'regression'
        
    print(f"Costruzione modello {args.model}...")
    model = FaceModel(
        model_name=args.model,
        mode=mode,
        num_classes=num_classes,
        pretrained=True
    )

    # 3. Avvio Training
    # Usiamo **vars(args) per passare gli argomenti della riga di comando come override alla config
    trainer = train_model(
        model=model,
        task=args.task,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        checkpoint_path=args.resume,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        freeze_epochs=args.freeze_epochs,
        scheduler_type=args.scheduler
    )
    
    print("\nTraining completato o interrotto.")

if __name__ == '__main__':
    main()