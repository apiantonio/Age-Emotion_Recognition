// False browser environment setup for Face-API and ONNX Runtime in a Web Worker
self.window = self;
self.document = {
    createElement: (type) => {
        if (type === 'canvas' || type === 'img') {
            const canvas = new OffscreenCanvas(1, 1);
            canvas.width = canvas.width || 1;
            canvas.height = canvas.height || 1;
            return canvas;
        }
        return {};
    }
};
self.HTMLVideoElement = class {};
self.HTMLImageElement = class {};
self.HTMLCanvasElement = OffscreenCanvas;
self.CanvasRenderingContext2D = OffscreenCanvasRenderingContext2D;

importScripts(
    'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.js'
);

faceapi.env.monkeyPatch({
    Canvas: OffscreenCanvas,
    Image: OffscreenCanvas,
    ImageData: ImageData,
    Video: class {},
    createCanvasElement: () => {
        const c = new OffscreenCanvas(1, 1);
        c.width = 1; c.height = 1;
        return c;
    },
    createImageElement: () => {
        const c = new OffscreenCanvas(1, 1);
        c.width = 1; c.height = 1;
        return c;
    }
});

let sessionEmo, sessionAge;
let isReady = false;

const EMO_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
const emoArray = new Float32Array(1 * 3 * 224 * 224);
const ageArray = new Float32Array(1 * 3 * 384 * 384);

function preprocess(imageData, targetSize, targetArray) {
    const data = imageData.data;
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const stride = targetSize * targetSize;

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        targetArray[j] = ((data[i] / 255.0) - mean[0]) / std[0];             // R
        targetArray[j + stride] = ((data[i + 1] / 255.0) - mean[1]) / std[1]; // G
        targetArray[j + stride * 2] = ((data[i + 2] / 255.0) - mean[2]) / std[2]; // B
    }
    return new ort.Tensor('float32', targetArray, [1, 3, targetSize, targetSize]);
}

async function initModels() {
    try {
        postMessage({ type: 'status', msg: 'Caricamento... ⏳' });

        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        ort.env.wasm.numThreads = 1;
        
        // Emozioni su CPU (per il modello quantizzato)
        const ortOptionsEmo = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
        // Età su GPU con fallback a CPU (per il modello Float32 originale)
        const ortOptionsAge = { executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' };
        
        // sessionEmo = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/emotion.onnx?download=true', ortOptionsEmo);
        sessionEmo = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/emotion_static_int8.onnx', ortOptionsEmo);

        sessionAge = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/age.onnx?download=true', ortOptionsAge);
        // sessionAge = await ort.InferenceSession.create('https://huggingface.co/datasets/apiantonio/facesight-models/resolve/main/age_static_int8.onnx', ortOptionsAge);

        // await faceapi.nets.ssdMobilenetv1.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/');
        await faceapi.nets.tinyFaceDetector.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/');
        
        isReady = true;
        postMessage({ type: 'ready' });
    } catch (e) {
        postMessage({ type: 'error', msg: e.message });
    }
}

// In ascolto dei comandi dal Main Thread
self.onmessage = async (e) => {
    if (e.data.type === 'init') {
        initModels();
    } else if (e.data.type === 'process' && isReady) {
        const { imageData, width, height } = e.data;
        
        const imgTensor = faceapi.tf.browser.fromPixels(imageData);
        const detections = await faceapi.detectAllFaces(imgTensor, new faceapi.TinyFaceDetectorOptions({ 
            inputSize: 416, scoreThreshold: 0.3 
        }));
        imgTensor.dispose();

        const results = [];
        
        if (detections && detections.length > 0) {
            // Disegna l'immagine ricevuta su un canvas invisibile per Face-API
            const canvas = new OffscreenCanvas(width, height);
            const ctx = canvas.getContext('2d');
            ctx.putImageData(imageData, 0, 0);
            
            const tmpCanvas = new OffscreenCanvas(384, 384);
            const tmpCtx = tmpCanvas.getContext('2d');
            const emoCanvas = new OffscreenCanvas(224, 224);
            const emoCtx = emoCanvas.getContext('2d');

            for (const detection of detections) {
                const box = detection.box;
                let padX = box.width * 0.2;
                let padY = box.height * 0.2;
                let x = Math.max(0, box.x - padX);
                let y = Math.max(0, box.y - padY);
                let w = Math.min(width - x, box.width + padX * 2);
                let h = Math.min(height - y, box.height + padY * 2);

                tmpCtx.clearRect(0, 0, 384, 384);
                tmpCtx.drawImage(canvas, x, y, w, h, 0, 0, 384, 384);

                emoCtx.clearRect(0, 0, 224, 224);
                emoCtx.drawImage(tmpCanvas, 0, 0, 384, 384, 0, 0, 224, 224);

                const emoTensor = preprocess(emoCtx.getImageData(0,0,224,224), 224, emoArray);
                const ageTensor = preprocess(tmpCtx.getImageData(0,0,384,384), 384, ageArray);

                const outEmo = await sessionEmo.run({ [sessionEmo.inputNames[0]]: emoTensor });
                const outAge = await sessionAge.run({ [sessionAge.inputNames[0]]: ageTensor });

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
                if (emotion === 'Angry')    color = '#FF0000'; 
                if (emotion === 'Surprise') color = '#00FFFF'; 
                if (emotion === 'Sad')      color = '#0040ff'; 
                if (emotion === 'Fear')     color = '#ff8000'; 

                results.push({ x, y, w, h, emotion, ageVal, color });
            }
        }
        
        // Rispediamo i calcoli finiti al Main Thread
        postMessage({ type: 'results', data: results });
    }
};