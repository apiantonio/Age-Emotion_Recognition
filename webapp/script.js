const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');

let worker;
let isWorkerBusy = false;
let latestResults = []; // Qui salva gli ultimi dati ricevuti dal worker

// Canvas nascosto per "fotografare" la webcam e inviarla al worker
const hiddenCanvas = document.createElement('canvas');
const hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

async function startSystem() {
    try {
        worker = new Worker('ai-worker.js');
        
        worker.onmessage = (e) => {
            if (e.data.type === 'status') {
                info.innerText = e.data.msg;
            } else if (e.data.type === 'ready') {
                info.innerText = "✅ Sistema pronto!";
                startWebcam();
            } else if (e.data.type === 'results') {
                latestResults = e.data.data; // Aggiorniamo i rettangoli
                isWorkerBusy = false;       
            } else if (e.data.type === 'error') {
                console.error("Worker Error:", e.data.msg);
                info.innerText = "❌ Errore IA: " + e.data.msg;
            }
        };

        worker.postMessage({ type: 'init' });

    } catch (e) {
        info.innerText = "❌ Errore sistema: " + e.message;
        console.error(e);
    }
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            video.play();
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            hiddenCanvas.width = video.videoWidth;
            hiddenCanvas.height = video.videoHeight;
            
            requestAnimationFrame(drawLoop);
        };
    } catch (err) {
        info.innerText = "❌ Errore Webcam. Permessi negati?";
        console.error("Dettaglio Errore Webcam:", err);
    }
}

function drawLoop() {
    if (!isWorkerBusy && video.readyState === video.HAVE_ENOUGH_DATA) {
        isWorkerBusy = true;
        
        hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        const imageData = hiddenCtx.getImageData(0, 0, hiddenCanvas.width, hiddenCanvas.height);
        
        worker.postMessage({ 
            type: 'process', 
            imageData: imageData, 
            width: hiddenCanvas.width, 
            height: hiddenCanvas.height 
        });
    }

    // A prescindere dallo stato del worker, disegna i rettangoli più recenti ricevuti
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (const face of latestResults) {
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

    requestAnimationFrame(drawLoop);
}

window.onload = startSystem;