document.addEventListener('DOMContentLoaded', () => {
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

    // Trigger file input on drop zone click
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    // Handle file drop
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // Handle file input change
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

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

        // UI Transitions
        previewContainer.classList.add('hidden');
        loading.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();

            if (response.ok) {
                loading.classList.add('hidden');
                resultSection.classList.remove('hidden');
                
                resultImage.src = data.image_base64;
                statusBadge.textContent = data.status;
                
                if (data.status === 'NORMAL') {
                    statusBadge.className = 'status-badge status-normal';
                } else {
                    statusBadge.className = 'status-badge status-rusak';
                }
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
});
