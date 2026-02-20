import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import os
import sys
from src.config import *


class BackboneController:
    """
    Gestisce il congelamento/scongelamento progressivo della backbone
    in modo AGNOSTICO rispetto all'architettura (ResNet, EfficientNet, ConvNeXt, ecc.).
    
    Invece di cercare nomi specifici come 'layer4', divide i parametri in base
    al loro ordine (i parametri finali sono sempre quelli più vicini all'output).
    """
    _current_stage = "init" 

    @staticmethod
    def set_trainable(model, trainable=False):
        """Imposta requires_grad per tutti i parametri della backbone"""
        for param in model.backbone.parameters():
            param.requires_grad = trainable

    @staticmethod
    def step(model, epoch, freeze_epochs=1):
        """
        Esegue lo step di unfreezing basato sull'epoca corrente.
        Strategia Universale:
        - Epoch < N:    Freeze Backbone (Solo Head)
        - Epoch == N:   Unfreeze ultimi 30% parametri (High level features)
        - Epoch == N+1: Unfreeze ultimi 60% parametri (Mid level features)
        - Epoch > N+1:  Unfreeze All (Fine tuning completo)
        """
        
        # Otteniamo tutti i parametri in una lista ordinata
        # In PyTorch, i parametri sono ordinati dall'input all'output.
        params = list(model.backbone.parameters())
        total_params = len(params)

        # FASE 1: FREEZE TOTALE (Warmup della Head)
        if epoch < freeze_epochs:
            if BackboneController._current_stage != "frozen":
                BackboneController.set_trainable(model, False)
                print(f"[Epoch {epoch+1}] Backbone FROZEN (Training Head Only)")
                BackboneController._current_stage = "frozen"

        # FASE 2: UNFREEZE PARZIALE (Ultimi 33% circa - High Level)
        elif epoch == freeze_epochs:
            if BackboneController._current_stage != "partial_high":
                # Resetta tutto a frozen
                BackboneController.set_trainable(model, False)
                
                # Scongela solo l'ultimo terzo dei parametri
                # (Equivale genericamente al blocco finale di conv)
                start_idx = int(total_params * 0.66) 
                
                for i, param in enumerate(params):
                    if i >= start_idx:
                        param.requires_grad = True

                print(f"[Epoch {epoch+1}] Unfreezing Top Layers (High-Level Features ~33%)")
                BackboneController._current_stage = "partial_high"

        # FASE 3: UNFREEZE MEDIO (Ultimi 66% circa - Mid Level)
        elif epoch == freeze_epochs * 2:
            if BackboneController._current_stage != "partial_mid":
                # Scongela gli ultimi due terzi
                start_idx = int(total_params * 0.33)
                
                for i, param in enumerate(params):
                    if i >= start_idx:
                        param.requires_grad = True
                        
                print(f"[Epoch {epoch+1}] Unfreezing Mid Layers (Mid-Level Features ~66%)")
                BackboneController._current_stage = "partial_mid"

        # FASE 4: FULL UNFREEZE
        else:
            if BackboneController._current_stage != "unfrozen":
                BackboneController.set_trainable(model, True)
                print(f"[Epoch {epoch+1}] Backbone FULLY UNFROZEN (Fine-Tuning All Layers)")
                BackboneController._current_stage = "unfrozen"


class MetricsTracker:
    """Traccia metriche durante il training"""

    def __init__(self, task='emotion'):
        self.task = task
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []

    def update(self, train_loss, val_loss, train_metric, val_metric, lr):
        # Ensure all values are standard Python floats before appending
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.train_metrics.append(float(train_metric))
        self.val_metrics.append(float(val_metric))
        self.learning_rates.append(float(lr))

    def get_metric_name(self):
        return "Accuracy" if self.task == 'emotion' else "MAE"

    def plot_history(self, save_path=None):
        """Genera grafici di training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Metric (Accuracy or MAE)
        metric_name = self.get_metric_name()
        axes[0, 1].plot(self.train_metrics, label=f'Train {metric_name}', linewidth=2)
        axes[0, 1].plot(self.val_metrics, label=f'Val {metric_name}', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel(metric_name)
        axes[0, 1].set_title(f'Training & Validation {metric_name}')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Learning Rate
        axes[1, 0].plot(self.learning_rates, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)

        # Overfitting Gap
        gap = np.array(self.val_losses) - np.array(self.train_losses)
        axes[1, 1].plot(gap, linewidth=2, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss - Train Loss')
        axes[1, 1].set_title('Overfitting Gap')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot salvato in: {save_path}")

        plt.show()

    def save_metrics(self, filepath):
        """Salva metriche in JSON"""
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates
        }
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metriche salvate in: {filepath}")


class EarlyStopping:
    """Early Stopping con patience e min_delta"""

    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"EARLY STOPPING attivato!")
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    def __init__(self, model, config, train_loader, val_loader, class_weights=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # Stato interno
        self.start_epoch = 0
        self.best_val_metric = float('-inf') if config.task == 'emotion' else float('inf')

        # Scaler per Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

        # Configurazione Loss
        if config.task == 'emotion':
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device), label_smoothing=config.label_smoothing)
            else:
                self.criterion = nn.CrossEntropyLoss()
            self.metric_mode = 'max'
            self.metric_name = 'Accuracy'
        else:
            self.criterion = nn.L1Loss(reduction='none')
            self.metric_mode = 'min'
            self.metric_name = 'MAE'

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler_type = getattr(config, 'scheduler', 'onecycle').lower()

        if self.scheduler_type == 'onecycle':
            print("Scheduler selezionato: OneCycleLR")
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1000.0
            )
        elif self.scheduler_type == 'plateau':
            print("Scheduler selezionato: ReduceLROnPlateau")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.metric_mode,
                factor=0.5,
                patience=3,
            )
        else:
            raise ValueError(f"Scheduler {self.scheduler_type} non supportato. Usa 'onecycle' o 'plateau'.")

        # Tracking
        self.metrics_tracker = MetricsTracker(task=config.task)
        self.early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta, mode=self.metric_mode)

    def save_checkpoint(self, epoch, metric, is_best=False, filename='last_checkpoint.pth'):
        """Salva lo stato completo del training"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_metric': self.best_val_metric,
            'config': self.config.__dict__
        }

        save_path = self.config.save_dir / filename
        torch.save(state, save_path)

        if is_best:
            best_path = self.config.save_dir / 'best_model.pth'
            torch.save(state, best_path)
            print(f"Checkpoint salvato: {filename} (Best: {is_best})")

    def load_checkpoint(self, checkpoint_path):
        """Carica un checkpoint per riprendere il training"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint non trovato in: {checkpoint_path}")
            return False

        print(f"Caricamento checkpoint da: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_metric = checkpoint['best_val_metric']

        print(f"Training ripristinato all'epoca {self.start_epoch}. Best Metric: {self.best_val_metric:.4f}")
        return True

    def train_epoch(self, epoch):
        print(f"\n{'='*70}")
        print(f"INIZIO EPOCH {epoch+1} [TRAIN]")
        sys.stdout.flush()

        self.model.train()
        BackboneController.step(self.model, epoch, freeze_epochs=self.config.freeze_epochs)

        running_loss = 0.0
        correct = 0; total = 0
        preds_all = []; targets_all = []

        print(f"Total batches: {len(self.train_loader)}")
        sys.stdout.flush()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [TRAIN]", disable=False)

        for batch_idx, batch in enumerate(pbar):
            try:
                if len(batch) == 3:
                    images, targets, sample_weights = batch
                    sample_weights = sample_weights.to(self.device, non_blocking=True)
                else:
                    images, targets = batch
                    sample_weights = None

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                    outputs = self.model(images)

                    if self.config.task == 'emotion':
                        loss = self.criterion(outputs, targets)
                    else:
                        outputs = outputs.squeeze()
                        targets = targets.float()
                        raw_loss = self.criterion(outputs, targets)
                        loss = (raw_loss * sample_weights).mean() if sample_weights is not None else raw_loss.mean()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                if self.scheduler_type == 'onecycle':
                    self.scheduler.step()

                running_loss += loss.item()
                if self.config.task == 'emotion':
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
                else:
                    preds_all.extend(outputs.detach().float().cpu().numpy())
                    targets_all.extend(targets.cpu().numpy())

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

            except Exception as e:
                print(f"\nERRORE al batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise

        avg_loss = running_loss / len(self.train_loader)
        if self.config.task == 'emotion':
            metric = 100.0 * correct / total
        else:
            metric = np.mean(np.abs(np.array(preds_all) - np.array(targets_all)))

        print(f"FINE EPOCH {epoch+1} [TRAIN] - Loss: {avg_loss:.4f}, {self.metric_name}: {metric:.4f}")
        sys.stdout.flush()

        return avg_loss, metric

    def validate(self, epoch):
        print(f"\n{'='*70}")
        print(f"INIZIO VALIDAZIONE [EPOCH {epoch+1}]")
        sys.stdout.flush()

        self.model.eval()
        running_loss = 0.0
        correct = 0; total = 0
        preds_all = []; targets_all = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [VAL]", disable=False)

            for batch_idx, batch in enumerate(pbar):
                try:
                    if len(batch) == 3:
                        images, targets, _ = batch
                    else:
                        images, targets = batch

                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                        outputs = self.model(images)

                        if self.config.task == 'emotion':
                            loss = self.criterion(outputs, targets)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == targets).sum().item()
                            total += targets.size(0)
                        else:
                            outputs = outputs.squeeze()
                            loss = self.criterion(outputs, targets.float()).mean()
                            preds_all.extend(outputs.float().cpu().numpy())
                            targets_all.extend(targets.cpu().numpy())

                    running_loss += loss.item()

                except Exception as e:
                    print(f"\nERRORE VAL al batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

        avg_loss = running_loss / len(self.val_loader)
        if self.config.task == 'emotion':
            metric = 100.0 * correct / total
        else:
            metric = np.mean(np.abs(np.array(preds_all) - np.array(targets_all)))

        print(f"FINE VALIDAZIONE [EPOCH {epoch+1}] - Loss: {avg_loss:.4f}, {self.metric_name}: {metric:.4f}")
        sys.stdout.flush()

        return avg_loss, metric

    def train(self):
        print(f"\n{'='*70}")
        print(f"START TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.config.device}")
        print(f"Scheduler: {self.scheduler_type.upper()}")
        print(f"AMP: {'ON' if self.config.use_amp else 'OFF'}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Checkpoints: {self.config.save_dir}")
        sys.stdout.flush()

    
        for epoch in range(self.start_epoch, self.config.num_epochs):
            t_loss, t_met = self.train_epoch(epoch)
            v_loss, v_met = self.validate(epoch)

            if self.scheduler_type == 'plateau':
                self.scheduler.step(v_met)

            self.metrics_tracker.update(t_loss, v_loss, t_met, v_met, self.optimizer.param_groups[0]['lr'])

            summary = f"Epoch {epoch+1}/{self.config.num_epochs} | Train Loss: {t_loss:.4f} {self.metric_name}: {t_met:.4f} | Val Loss: {v_loss:.4f} {self.metric_name}: {v_met:.4f}"
            print(f"\n{summary}")
            sys.stdout.flush()

            if self.metric_mode == 'max':
                is_best = v_met > self.best_val_metric
            else:
                is_best = v_met < self.best_val_metric

            if is_best:
                self.best_val_metric = v_met
                print(f"New Best Metric: {self.best_val_metric:.4f}")
                sys.stdout.flush()

            self.save_checkpoint(epoch, v_met, is_best=is_best, filename='last_checkpoint.pth')

            self.early_stopping(v_met)
            if self.early_stopping.early_stop:
                print("\nEarly Stopping attivato.")
                sys.stdout.flush()
                break

        self.metrics_tracker.save_metrics(self.config.save_dir / 'metrics.json')
        self.metrics_tracker.plot_history(self.config.save_dir / 'history.png')
        
def train_model(model, task, train_loader, val_loader,
                class_weights=None,
                checkpoint_path=None,
                **config_kwargs):
    """
    Funzione wrapper robusta per avviare il training.

    Args:
        model: Il modello da addestrare (FaceModel).
        task: 'emotion' o 'age'.
        train_loader: DataLoader di training.
        val_loader: DataLoader di validazione.
        class_weights: (Opzionale) Tensore dei pesi per CrossEntropy (solo task emotion).
        checkpoint_path: (Opzionale) Percorso del file .pth da caricare per riprendere il training.
        **config_kwargs: Argomenti opzionali per sovrascrivere la config (es. num_epochs=50).

    Returns:
        trainer: L'istanza del Trainer addestrata.
    """
    config = TrainingConfig(task=task)

    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"Config Override: {key} = {value}")

    model = model.to(config.device)

    weights_to_pass = class_weights if task == 'emotion' else None

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=weights_to_pass,
    )

    # resume training
    if checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrotto manualmente dall'utente.")

    return trainer