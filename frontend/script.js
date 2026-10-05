document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audio-upload');
    const predictBtn = document.getElementById('predict-btn');
    const loadingBox = document.getElementById('loading');
    const resultBox = document.getElementById('result-box');
    const predictionText = document.getElementById('prediction-text');

    predictBtn.addEventListener('click', async () => {
        if (!fileInput.files.length) {
            alert("Please select a .wav file first.");
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('audio', file);
        
        resultBox.classList.add('hidden');
        loadingBox.classList.remove('hidden');
        predictBtn.disabled = true;
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            loadingBox.classList.add('hidden');
            
            if (response.ok) {
                predictionText.innerText = data.prediction;
                resultBox.classList.remove('hidden');
            } else {
                alert("Error: " + data.error);
            }
        } catch (error) {
            loadingBox.classList.add('hidden');
            alert("Network error occurred.");
        } finally {
            predictBtn.disabled = false;
        }
    });
});
