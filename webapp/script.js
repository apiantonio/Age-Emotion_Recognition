const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const info = document.getElementById('info');

const EMO_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
let sessionEmo, sessionAge;

let isProcessing = false; 
let lastProcessTime = 0;
let dynamicInterval = 0; // Base ms
const MIN_INTERVAL = 200;  // Velocità massima per telefoni top di gamma (30+ FPS)
const MAX_INTERVAL = 10000; // Lentezza massima per telefoni di fascia bassa (1 FPS)

async function initSystem() {
    try {
        if (navigator.gpu) {
            console.log("WebGPU detected, using it for ONNX Runtime. This should provide better performance on supported devices.");
        } else {
            console.warn("WebGPU not supported, using fallback to WebAssembly. Performance may be reduced on some devices.");
        }
        info.innerText = "Caricamento... ⏳";
        
        // cores optimized for mobile devices
        const numCores = navigator.hardwareConcurrency || 4;
        ort.env.wasm.numThreads = Math.min(numCores, 4);
        
        const ortOptions = { 
            executionProviders: ['wasm'], //['webgpu', 'wasm'], // provare WebGL prima di WASM, solo WASM con modelli quantizzati
            graphOptimizationLevel: 'all'
        };

        // sessionEmo = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/emotion.onnx?download=true', ortOptions);
        sessionEmo = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/emotion_static_int8.onnx', ortOptions);

        // sessionAge = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/age.onnx?download=true', ortOptions);
        sessionAge = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/age_static_int8.onnx', ortOptions);

        // await faceapi.nets.ssdMobilenetv1.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/');
        await faceapi.nets.tinyFaceDetector.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/');
        
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

const emoArray = new Float32Array(3 * 224 * 224);
const ageArray = new Float32Array(3 * 384 * 384);

// Preprocess function: normalizza e riorganizza i dati in formato CHW
function preprocess(imgData, targetSize, float32Data) {
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

const tmpCanvas = document.createElement('canvas');
tmpCanvas.width = 384; tmpCanvas.height = 384;
const tmpCtx = tmpCanvas.getContext('2d', { willReadFrequently: true });

const emoCanvas = document.createElement('canvas');
emoCanvas.width = 224; emoCanvas.height = 224;
const emoCtx = emoCanvas.getContext('2d', { willReadFrequently: true });

async function processFrame() {
    const now = Date.now();

    if (now - lastProcessTime >= dynamicInterval) {
        if (!isProcessing && sessionEmo && sessionAge) {
            isProcessing = true; 
            lastProcessTime = now;
            const startTime = performance.now();

            try {
                // face-api
                // const detections = await faceapi.detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }));
                const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 }));   
                console.log("Volti rilevati:", detections ? detections.length : 0);
                const facesData = [];
                
                if (detections && detections.length > 0) {
                    for (const detection of detections) {
                        const box = detection.box;
                        
                        let padX = box.width * 0.2;
                        let padY = box.height * 0.2;
                        let x = Math.max(0, box.x - padX);
                        let y = Math.max(0, box.y - padY);
                        let w = Math.min(canvas.width - x, box.width + padX * 2);
                        let h = Math.min(canvas.height - y, box.height + padY * 2);

                        /* Debug */
                        // ctx.strokeStyle = "white";
                        // ctx.lineWidth = 3;
                        // ctx.strokeRect(x, y, w, h);
                        
                        ctx.lineWidth = 3;
                        tmpCtx.clearRect(0, 0, 384, 384);
                        tmpCtx.drawImage(video, x, y, w, h, 0, 0, 384, 384);

                        emoCtx.clearRect(0, 0, 224, 224);
                        emoCtx.drawImage(tmpCanvas, 0, 0, 384, 384, 0, 0, 224, 224);
                        
                        const emoTensor = preprocess(emoCtx.getImageData(0,0,224,224), 224, emoArray);
                        const ageTensor = preprocess(tmpCtx.getImageData(0,0,384,384), 384, ageArray);

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
                        if (emotion === 'Neutral')  color = '#FFFF00'; 
                        if (emotion === 'Disgust')  color = '#800080'; 
                        if (emotion === 'Happy')    color = '#00FF00'; 
                        if (emotion === 'Angry')    color = '#FF0000'; 
                        if (emotion === 'Surprise') color = '#00FFFF'; 
                        if (emotion === 'Sad')      color = '#0040ff'; 
                        if (emotion === 'Fear')     color = '#ff8000'; 

                        facesData.push({ x, y, w, h, emotion, ageVal, color });
                    }
                }

                // Clear canvas and draw results
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                for (const face of facesData) {
                    ctx.strokeStyle = face.color;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(face.x, face.y, face.w, face.h);

                    const text = `${face.emotion} | Età: ${face.ageVal.toFixed(1)}`;
                    ctx.font = 'bold 22px Arial';
                    const textWidth = ctx.measureText(text).width;

                    ctx.save();
                    ctx.scale(-1, 1); 

                    const textX = -(face.x + face.w); 
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                    ctx.fillRect(textX, face.y - 32, textWidth + 10, 32);
                    
                    ctx.fillStyle = face.color;
                    ctx.fillText(text, textX + 5, face.y - 8);

                    ctx.restore(); 
                }

            } catch (error) {
                console.error("ONNX Inference Error:", error);
            }

        const endTime = performance.now();
        const timeTaken = endTime - startTime;

        // adapts the next interval based on how long the processing took, aiming for a balance between responsiveness and performance
        let targetInterval = timeTaken * 1.5; 
        dynamicInterval = Math.max(MIN_INTERVAL, Math.min(MAX_INTERVAL, targetInterval));

        console.log(`Compute time: ${timeTaken.toFixed(0)}ms | Next interval: ${dynamicInterval.toFixed(0)}ms`);

        isProcessing = false; 
    }
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