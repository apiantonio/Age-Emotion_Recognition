import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
from src.config import MODELS_CONFIG

# --- TRUCCO PER L'ESEGUIBILE ---
# Quando diventerà un .exe, i path cambiano. Questa funzione trova sempre i file.
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 1. CARICAMENTO MODELLI ONNX
print("Caricamento motori ONNX in corso...")
emo_path = get_resource_path("./onnx/emotion.onnx")
age_path = get_resource_path("./onnx/age.onnx")

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]

session_emo = ort.InferenceSession(emo_path, providers=providers)
session_age = ort.InferenceSession(age_path, providers=providers)
print(f"Dispositivo in uso (Emo): {session_emo.get_providers()[0]}")

EMO_LABELS = MODELS_CONFIG['emotion']['labels']

# INIZIALIZZAZIONE FACE DETECTOR (Senza PyTorch)
# Usiamo il rilevatore nativo di OpenCV. È leggero e non richiede librerie esterne.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FUNZIONE DI PREPROCESSING (Sostituisce torchvision.transforms)
def preprocess_image(img_bgr, resize_dim, crop_dim):
    # Da BGR a RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize e Crop
    img_resized = cv2.resize(img_rgb, (resize_dim, resize_dim))
    start = resize_dim // 2 - crop_dim // 2
    img_cropped = img_resized[start:start+crop_dim, start:start+crop_dim]
    
    # Normalizzazione (ToTensor + Normalize)
    img_float = img_cropped.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    
    # Da HWC a CHW (Channels First)
    img_chw = np.transpose(img_norm, (2, 0, 1))
    
    # Aggiungi dimensione batch
    return np.expand_dims(img_chw, axis=0)

# 4. LOOP PRINCIPALE
def main():
    cap = cv2.VideoCapture(0)
    print("Applicazione avviata! Premi 'Q' per uscire.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trova facce
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            # Padding
            pad = int(h * 0.2)
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0: continue

            # Prepara i tensori per ONNX
            input_emo = preprocess_image(face_img, 236, 224)
            input_age = preprocess_image(face_img, 384, 384)

            # Esegui l'inferenza
            out_emo = session_emo.run(None, {'input': input_emo})[0]
            out_age = session_age.run(None, {'input': input_age})[0]

            # Post-processing Emozione (Softmax)
            emo_logits = out_emo[0]
            exp_preds = np.exp(emo_logits - np.max(emo_logits))
            probs = exp_preds / np.sum(exp_preds)
            emo_idx = np.argmax(probs)
            emo_conf = probs[emo_idx]
            emo_label = EMO_LABELS[emo_idx]

            # Post-processing Età
            age_val = float(out_age[0][0])
            age_val = max(1, min(100, age_val))

            # Disegno UI
            color = (0, 255, 0)
            if emo_label in ['Sad', 'Fear', 'Angry']: color = (0, 0, 255)
            elif emo_label in ['Happy', 'Surprise']: color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            testo = f"{emo_label} ({emo_conf:.2f}) | Age: {age_val:.1f}"
            cv2.putText(frame, testo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('FaceSight - LITE (.exe version)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()