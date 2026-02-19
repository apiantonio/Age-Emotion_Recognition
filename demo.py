import os
import sys
import cv2
import time
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import pathlib
sys.modules['pathlib._local'] = pathlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import FaceModel
from src.config import *
from src.face_stabilizer import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

# Impostazioni Webcam
WEBCAM_ID = 0
FRAME_WIDTH = 1080
FRAME_HEIGHT = 720
STABILITY_FACTOR = 0.1  # (0.1 molto stabile, 0.5 medio)
FRAME_SKIP = 2 # processa 1 frame ogni skip+1 (es. 4 = processa 1 frame ogni 5)
MIN_FACE_SIZE = 80
USE_FP16 = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device usato: {device}")

def get_transform(resize_dim, crop_dim):
    return transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.CenterCrop((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_model_from_config(task_name):
    cfg = MODELS_CONFIG[task_name]
    print(f"Caricamento {task_name.upper()} ({cfg['model_name']})...")
    
    mode = 'classification' if task_name == 'emotion' else 'regression'
    model = FaceModel(model_name=cfg['model_name'], mode=mode, num_classes=cfg['num_classes'], pretrained=False)
    
    if not os.path.exists(cfg['checkpoint']):
        print(f"ERRORE: Checkpoint non trovato: {cfg['checkpoint']}")
        return None, None

    # Aggiunto weights_only=False per evitare warning/errori
    checkpoint = torch.load(cfg['checkpoint'], map_location=device, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    if USE_FP16 and device.type == 'cuda':
        model.half() # Converte i pesi in float16
        print("   Modalità FP16 attivata.")
    
    transform = get_transform(cfg['resize'], cfg['crop'])
    
    return model, transform


# Caricamento Modelli Task
model_emo, transform_emo = load_model_from_config('emotion')
model_age, transform_age = load_model_from_config('age')

if model_emo is None or model_age is None:
    sys.exit(1)

# CUDA Streams
if device.type == 'cuda':
    stream_emo = torch.cuda.Stream()
    stream_age = torch.cuda.Stream()
    print("CUDA Streams pronti.")
else:
    stream_emo = None; stream_age = None

# --- SETUP MTCNN ---
print("Inizializzazione MTCNN...")
# keep_all=True: rileva tutte le facce, non solo la più grande
# device=device: usa la GPU per il rilevamento (fondamentale per velocità)
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=MIN_FACE_SIZE, thresholds=[0.6, 0.7, 0.7])
print("MTCNN pronto.")

# --- INIZIALIZZAZIONE STABILIZZATORE ---
stabilizer = FaceStabilizer(alpha=STABILITY_FACTOR)
print(f"Stabilizzazione attiva (Fattore: {STABILITY_FACTOR})")

def main():
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    prev_time = 0
    frame_count = 0

    cached_coords = []
    cached_ages = []
    cached_emos = []
    cached_confs = []

    print("\n Avvio stream. Premi 'Q' per uscire.\n")

    with torch.no_grad():
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            # Mirroring
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
        
            # Se siamo nel frame "giusto", ricalcoliamo tutto.
            # Altrimenti usiamo i dati vecchi (cached)
            if frame_count % (FRAME_SKIP + 1) == 0:
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                # 1. MTCNN Detection
                boxes, _ = mtcnn.detect(pil_frame)
                
                crops_emo = []; crops_age = []; new_coords = []

                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        bw, bh = x2 - x1, y2 - y1

                        # Padding
                        pad = int(bh * 0.2)
                        x_p = max(0, x1 - pad); y_p = max(0, y1 - pad)
                        bw_p = min(w - x_p, bw + 2*pad); bh_p = min(h - y_p, bh + 2*pad)
                        
                        face_img = rgb_frame[y_p:y_p+bh_p, x_p:x_p+bw_p]
                        if face_img.size == 0: continue

                        pil_face = Image.fromarray(face_img)
                        crops_emo.append(transform_emo(pil_face))
                        crops_age.append(transform_age(pil_face))
                        new_coords.append((x1, y1, bw, bh))

                # Se abbiamo trovato facce, facciamo inferenza
                if crops_emo:
                    # Creazione Batch
                    batch_emo = torch.stack(crops_emo).to(device)
                    batch_age = torch.stack(crops_age).to(device)

                    # Conversione FP16 se attiva
                    if USE_FP16 and device.type == 'cuda':
                        batch_emo = batch_emo.half()
                        batch_age = batch_age.half()

                    # Inferenza Parallela
                    if device.type == 'cuda':
                        with torch.cuda.stream(stream_emo):
                            out_emo = model_emo(batch_emo)
                            probs = torch.softmax(out_emo, dim=1)
                            new_emos_idx = torch.argmax(probs, dim=1).cpu().numpy()
                            new_confs = torch.max(probs, dim=1).values.cpu().numpy()

                        with torch.cuda.stream(stream_age):
                            out_age = model_age(batch_age)
                            raw_ages = out_age.cpu().numpy().flatten()
                        
                        torch.cuda.synchronize()
                    else:
                        # Fallback CPU
                        out_emo = model_emo(batch_emo) # CPU non supporta half() solitamente bene come CUDA
                        probs = torch.softmax(out_emo, dim=1)
                        new_emos_idx = torch.argmax(probs, dim=1).cpu().numpy()
                        new_confs = torch.max(probs, dim=1).values.cpu().numpy()
                        out_age = model_age(batch_age)
                        raw_ages = out_age.cpu().numpy().flatten()

                    # Stabilizzazione
                    new_stable_ages = stabilizer.update(new_coords, raw_ages)

                    # AGGIORNIAMO LA CACHE
                    cached_coords = new_coords
                    cached_ages = new_stable_ages
                    cached_emos = new_emos_idx
                    cached_confs = new_confs
                else:
                    # Nessuna faccia trovata
                    cached_coords = []
                    cached_ages = []
                    cached_emos = []
                    cached_confs = []

            frame_count += 1
            labels_map = MODELS_CONFIG['emotion']['labels']
            
            for i, (fx, fy, fw, fh) in enumerate(cached_coords):
                # Dati dalla cache
                if i >= len(cached_ages): break # Sicurezza

                age = float(cached_ages[i])
                age = max(1, min(100, age))
                
                emo_lbl = labels_map[cached_emos[i]]
                emo_conf = cached_confs[i]

                # Colore box
                # Colori
                color = (0, 255, 0)
                # 'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
                if emo_lbl == 'Angry': color = (0, 0, 255)
                elif emo_lbl == 'Disgust': color = (0, 255, 0)
                elif emo_lbl == 'Fear': color = (255, 255, 0)
                elif emo_lbl == 'Happy': color = (0, 255, 255)
                elif emo_lbl == 'Neutral': color = (255, 255, 0)
                elif emo_lbl == 'Sad': color = (255, 0, 0)
                elif emo_lbl == 'Surprise': color = (255, 0, 255)

                # Rettangolo
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)

                # Info Text
                info_text = f"{emo_lbl} ({emo_conf:.2f}) | Age: {age:.1f}"
                (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Sfondo semitrasparente per il testo (più bello)
                overlay = frame.copy()
                cv2.rectangle(overlay, (fx, fy - 25), (fx + tw + 10, fy), color, -1)
                alpha = 0.7 # Trasparenza
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                cv2.putText(frame, info_text, (fx + 5, fy - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

            # FPS Counter
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Age & Emotion - FAST', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()