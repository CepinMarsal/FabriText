document.addEventListener('DOMContentLoaded', () => {
    // --- TAB LOGIC ---
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active', 'hidden'));
            tabContents.forEach(c => {
                if(c.id !== targetId) c.classList.add('hidden');
            });

            // Add active class
            btn.classList.add('active');
            document.getElementById(targetId).classList.add('active');

            // If switching away from camera tab, stop the camera
            if (targetId !== 'camera-tab') {
                stopCamera();
            }
            if (targetId !== 'lbp-tab') {
                stopLbpCamera();
            }
        });
    });

    // --- UPLOAD LOGIC ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    const loading = document.getElementById('loading');
    
    const resultSection = document.getElementById('result-section');
    const statusBadge = document.getElementById('status-badge');
    const resultImage = document.getElementById('result-image');
    const tryAgainBtn = document.getElementById('try-again-btn');

    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                selectedFile = file;
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => {
                    imagePreview.src = reader.result;
                    dropZone.classList.add('hidden');
                    previewContainer.classList.remove('hidden');
                }
            } else {
                alert('Tolong unggah file gambar yang valid.');
            }
        }
    }

    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    tryAgainBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        resetBtn.click();
    });

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        previewContainer.classList.add('hidden');
        loading.classList.remove('hidden');

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok) {
                loading.classList.add('hidden');
                resultSection.classList.remove('hidden');
                resultImage.src = data.image_base64;
                statusBadge.textContent = data.status;
                statusBadge.className = data.status === 'NORMAL' ? 'status-badge status-normal' : 'status-badge status-rusak';
            } else {
                alert('Error: ' + (data.error || 'Terjadi kesalahan'));
                loading.classList.add('hidden');
                previewContainer.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Gagal terhubung ke server');
            loading.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        }
    });

    // --- CAMERA LOGIC ---
    const cameraSelect = document.getElementById('camera-select');
    const modeSelect = document.getElementById('mode-select');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const stopCameraBtn = document.getElementById('stop-camera-btn');
    const cameraContainer = document.getElementById('camera-container');
    const videoElement = document.getElementById('video');
    const canvasElement = document.getElementById('canvas');
    const cameraFeed = document.getElementById('camera-feed');
    const liveStatusBadge = document.getElementById('live-status-badge');
    const confMetric = document.getElementById('conf-metric');

    let stream = null;
    let cameraInterval = null;
    let isProcessing = false;

    // Enumerate kamera saat halaman load
    async function populateCameras() {
        try {
            await navigator.mediaDevices.getUserMedia({ video: true }).then(s => s.getTracks().forEach(t => t.stop()));
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');
            cameraSelect.innerHTML = '';
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Kamera ${index + 1}`;
                cameraSelect.appendChild(option);
            });
        } catch (err) {
            console.error('Gagal enumerate kamera', err);
            const option = document.createElement('option');
            option.text = 'Kamera tidak ditemukan / tidak diizinkan';
            cameraSelect.appendChild(option);
        }
    }

    populateCameras();

    async function startCamera() {
        const deviceId = cameraSelect.value;
        const constraints = {
            video: deviceId
                ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
                : { width: { ideal: 640 }, height: { ideal: 480 } }
        };

        try {
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = stream;

            startCameraBtn.classList.add('hidden');
            stopCameraBtn.classList.remove('hidden');
            cameraContainer.classList.remove('hidden');

            videoElement.onloadedmetadata = () => {
                videoElement.play();
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                cameraInterval = setInterval(processFrame, 500);
            };
        } catch (err) {
            console.error('Gagal memulai kamera', err);
            alert('Gagal memulai kamera: ' + err.message);
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        if (cameraInterval) {
            clearInterval(cameraInterval);
            cameraInterval = null;
        }
        
        startCameraBtn.classList.remove('hidden');
        stopCameraBtn.classList.add('hidden');
        cameraContainer.classList.add('hidden');
    }

    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);

    async function processFrame() {
        if (isProcessing || !stream) return;
        isProcessing = true;

        try {
            const ctx = canvasElement.getContext('2d');
            ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            const frameBase64 = canvasElement.toDataURL('image/jpeg', 0.8);

            const endpoint = modeSelect.value === 'fixed' ? '/predict_frame_fixed' : '/predict_frame';
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameBase64 })
            });

            if (response.ok) {
                const data = await response.json();
                cameraFeed.src = data.image_base64;
                liveStatusBadge.textContent = data.status;
                liveStatusBadge.className = data.status === 'NORMAL' ? 'status-badge status-normal' : 'status-badge status-rusak';
                confMetric.textContent = `Conf: ${data.confidence}`;
            }
        } catch (err) {
            console.error("Frame processing error:", err);
        } finally {
            isProcessing = false;
        }
    }

    // --- LBP CAMERA LOGIC ---
    const lbpCameraSelect = document.getElementById('lbp-camera-select');
    const startLbpBtn = document.getElementById('start-lbp-btn');
    const stopLbpBtn = document.getElementById('stop-lbp-btn');
    const lbpContainer = document.getElementById('lbp-container');
    const lbpVideo = document.getElementById('lbp-video');
    const lbpCanvas = document.getElementById('lbp-canvas');
    const lbpFeed = document.getElementById('lbp-feed');
    const lbpStatusBadge = document.getElementById('lbp-status-badge');
    const lbpConfMetric = document.getElementById('lbp-conf-metric');

    let lbpStream = null;
    let lbpInterval = null;
    let lbpProcessing = false;

    async function startLbpCamera() {
        const deviceId = lbpCameraSelect.value;
        const constraints = {
            video: deviceId
                ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
                : { width: { ideal: 640 }, height: { ideal: 480 } }
        };
        try {
            lbpStream = await navigator.mediaDevices.getUserMedia(constraints);
            lbpVideo.srcObject = lbpStream;
            startLbpBtn.classList.add('hidden');
            stopLbpBtn.classList.remove('hidden');
            lbpContainer.classList.remove('hidden');
            lbpVideo.onloadedmetadata = () => {
                lbpVideo.play();
                lbpCanvas.width = lbpVideo.videoWidth;
                lbpCanvas.height = lbpVideo.videoHeight;
                lbpInterval = setInterval(processLbpFrame, 500);
            };
        } catch (err) {
            console.error('Gagal memulai kamera LBP', err);
            alert('Gagal memulai kamera: ' + err.message);
        }
    }

    function stopLbpCamera() {
        if (lbpStream) {
            lbpStream.getTracks().forEach(t => t.stop());
            lbpStream = null;
        }
        if (lbpInterval) {
            clearInterval(lbpInterval);
            lbpInterval = null;
        }
        startLbpBtn.classList.remove('hidden');
        stopLbpBtn.classList.add('hidden');
        lbpContainer.classList.add('hidden');
    }

    startLbpBtn.addEventListener('click', startLbpCamera);
    stopLbpBtn.addEventListener('click', stopLbpCamera);

    async function processLbpFrame() {
        if (lbpProcessing || !lbpStream) return;
        lbpProcessing = true;
        try {
            const ctx = lbpCanvas.getContext('2d');
            ctx.drawImage(lbpVideo, 0, 0, lbpCanvas.width, lbpCanvas.height);
            const frameBase64 = lbpCanvas.toDataURL('image/jpeg', 0.8);

            const response = await fetch('/predict_frame_lbp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameBase64 })
            });

            if (response.ok) {
                const data = await response.json();
                lbpFeed.src = data.image_base64;
                lbpStatusBadge.textContent = data.status;
                lbpStatusBadge.className = data.status === 'NORMAL' ? 'status-badge status-normal' : 'status-badge status-rusak';
                lbpConfMetric.textContent = `Conf: ${data.confidence}`;
            }
        } catch (err) {
            console.error('LBP frame error:', err);
        } finally {
            lbpProcessing = false;
        }
    }

    // Populate LBP camera dropdown saat tab LBP dibuka
    document.querySelector('[data-target="lbp-tab"]').addEventListener('click', async () => {
        stopLbpCamera();
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');
            lbpCameraSelect.innerHTML = '';
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Kamera ${index + 1}`;
                lbpCameraSelect.appendChild(option);
            });
        } catch (err) {
            console.error('Gagal enumerate kamera LBP', err);
        }
    });
});
