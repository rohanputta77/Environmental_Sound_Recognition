document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audio-upload');
    const predictBtn = document.getElementById('predict-btn');
    const loadingBox = document.getElementById('loading');
    const resultBox = document.getElementById('result-box');
    const predictionText = document.getElementById('prediction-text');
    const confBar = document.getElementById('conf-bar');
    const confValue = document.getElementById('conf-value');
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');

    // Canvas scaling for high DPI displays
    function resizeCanvas() {
        canvas.width = canvas.clientWidth * window.devicePixelRatio;
        canvas.height = canvas.clientHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    let audioContext;

    function drawWaveform(buffer) {
        const rawData = buffer.getChannelData(0); 
        const samples = Math.floor(canvas.clientWidth); 
        const blockSize = Math.floor(rawData.length / samples);
        
        // Use client width/height for logical drawing space
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#ffb000'; // Accent color
        
        const centerY = height / 2;
        const heightMultiplier = centerY * 0.9;

        for (let i = 0; i < samples; i++) {
            let min = 1.0;
            let max = -1.0;
            for (let j = 0; j < blockSize; j++) {
                const datum = rawData[(i * blockSize) + j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }
            
            const x = i;
            const y = centerY + (min * heightMultiplier);
            const drawHeight = Math.max(1, (max - min) * heightMultiplier);
            
            ctx.fillRect(x, y, 1, drawHeight);
        }
    }

    fileInput.addEventListener('change', async () => {
        if (fileInput.files.length === 0) return;
        
        // Initialize context on user interaction
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        const file = fileInput.files[0];
        const arrayBuffer = await file.arrayBuffer();
        
        audioContext.decodeAudioData(arrayBuffer, (buffer) => {
            resizeCanvas(); 
            drawWaveform(buffer);
        });
    });

    predictBtn.addEventListener('click', async () => {
        if (!fileInput.files.length) {
            alert("NO SOURCE FILE DETECTED");
            return;
        }
        
        const file = fileInput.files[0];
        const modelName = document.getElementById('model-select').value;
        const formData = new FormData();
        formData.append('audio', file);
        formData.append('model_name', modelName);
        
        resultBox.classList.add('hidden');
        loadingBox.classList.remove('hidden');
        predictBtn.disabled = true;
        predictBtn.innerText = "PROCESSING...";
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            loadingBox.classList.add('hidden');
            
            if (response.ok) {
                predictionText.innerText = data.prediction;
                
                // Read exact mathematical confidence directly from the AI backend
                const realConf = data.confidence.toFixed(1);
                
                resultBox.classList.remove('hidden');
                
                // Trigger reflow for animation
                confBar.style.width = '0%';
                setTimeout(() => {
                    confBar.style.width = realConf + '%';
                    confValue.innerText = realConf + '%';
                }, 50);
                
            } else {
                alert("ERR: " + data.error);
            }
        } catch (error) {
            loadingBox.classList.add('hidden');
            alert("SYSTEM ERROR: NETWORK FAILURE");
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerText = "EXECUTE ANALYSIS";
        }
    });
});
