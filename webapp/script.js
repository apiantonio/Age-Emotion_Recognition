const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const info = document.getElementById('info');

const EMO_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
let sessionEmo, sessionAge;

let isProcessing = false; 

async function initSystem() {
    try {
        info.innerText = "Caricamento... ⏳";
        
        const ortOptions = { 
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };
        sessionEmo = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/emotion.onnx?download=true', ortOptions);
        sessionAge = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/age.onnx?download=true', ortOptions);

        info.innerText = "Caricamento... ⏳";
        
        await faceapi.nets.ssdMobilenetv1.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/');

        info.innerText = "✅ Sistema pronto!";
        startWebcam();
    } catch (e) {
        info.innerText = "❌ Errore: " + e.message;
        console.error(e);
    }
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            video.play();
            requestAnimationFrame(processFrame);
        };
    } catch (e) {
        info.innerText = "❌ Errore webcam: " + e.message;
    }
}

function preprocess(imgData, targetSize) {
    const float32Data = new Float32Array(3 * targetSize * targetSize);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    let dataIndex = 0;
    
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < targetSize * targetSize; i++) {
            float32Data[dataIndex++] = ((imgData.data[i * 4 + c] / 255.0) - mean[c]) / std[c];
        }
    }
    return new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]);
}

async function processFrame() {
    if (!isProcessing && sessionEmo && sessionAge) {
        
        isProcessing = true; 
        
        try {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // face-api
            const detections = await faceapi.detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }));

            if (detections && detections.length > 0) {
                for (const detection of detections) {
                    const box = detection.box;
                    
                    let padX = box.width * 0.2;
                    let padY = box.height * 0.2;
                    let x = Math.max(0, box.x - padX);
                    let y = Math.max(0, box.y - padY);
                    let w = Math.min(canvas.width - x, box.width + padX * 2);
                    let h = Math.min(canvas.height - y, box.height + padY * 2);

                    ctx.strokeStyle = "white";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, w, h);

                    const tmpCanvas = document.createElement('canvas');
                    tmpCanvas.width = 384; tmpCanvas.height = 384;
                    const tmpCtx = tmpCanvas.getContext('2d', { willReadFrequently: true });
                    tmpCtx.drawImage(video, x, y, w, h, 0, 0, 384, 384);

                    const emoCanvas = document.createElement('canvas');
                    emoCanvas.width = 224; emoCanvas.height = 224;
                    emoCanvas.getContext('2d').drawImage(tmpCanvas, 0, 0, 384, 384, 0, 0, 224, 224);
                    
                    const emoTensor = preprocess(emoCanvas.getContext('2d').getImageData(0,0,224,224), 224);
                    const ageTensor = preprocess(tmpCtx.getImageData(0,0,384,384), 384);

                    const outEmo = await sessionEmo.run({ input: emoTensor });
                    const outAge = await sessionAge.run({ input: ageTensor });

                    const logits = outEmo.output.data;
                    const maxLogit = Math.max(...logits);
                    const expLogits = Array.from(logits).map(val => Math.exp(val - maxLogit));
                    const sumExp = expLogits.reduce((a, b) => a + b);
                    const probs = expLogits.map(val => val / sumExp);
                    const maxIndex = probs.indexOf(Math.max(...probs));
                    const emotion = EMO_LABELS[maxIndex];

                    let ageVal = Math.max(1, Math.min(100, outAge.output.data[0]));

                    let color = '#00FF00'; 
                    if (emotion === 'Neutral') color = '#FFFF00'; 
                    if (emotion === 'Disgust') color = '#800080'; 
                    if (emotion === 'Happy') color = '#00FF00'; 
                    if (emotion === 'Angry') color = '#FF0000'; 
                    if (emotion === 'Surprise') color = '#00FFFF'; 
                    if (emotion === 'Sad') color = '#0040ff'; 
                    if (emotion === 'Fear') color = '#ff8000'; 

                    ctx.strokeStyle = color;
                    ctx.strokeRect(x, y, w, h);

                    const text = `${emotion} | Età: ${ageVal.toFixed(1)}`;
                    ctx.font = 'bold 22px Arial';
                    const textWidth = ctx.measureText(text).width;

                    ctx.save();
                    ctx.scale(-1, 1); 

                    const textX = -(x + w); 
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                    ctx.fillRect(textX, y - 32, textWidth + 10, 32);
                    
                    ctx.fillStyle = color;
                    ctx.fillText(text, textX + 5, y - 8);

                    ctx.restore(); 
                }
            }
        } catch (error) {
            console.error("ONNX Inference Error:", error);
        }

        isProcessing = false; 
    }

    requestAnimationFrame(processFrame);
}

initSystem();

if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js')
            .then(reg => console.log('Service Worker registrato con successo!'))
            .catch(err => console.error('Errore registrazione SW:', err));
    });
}