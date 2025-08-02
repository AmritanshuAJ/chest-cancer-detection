// Medical Interface JavaScript with Base64 Support
document.addEventListener('DOMContentLoaded', function() {
    initializeInterface();
});

let currentImageBase64 = null;

function initializeInterface() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const fileDetails = document.getElementById('fileDetails');
    const analysisForm = document.getElementById('analysisForm');
    const loadingOverlay = document.getElementById('loadingOverlay');

    // Initialize fade-in animations
    const elements = document.querySelectorAll('.fade-in');
    elements.forEach((el, index) => {
        el.style.animationDelay = `${index * 0.1}s`;
    });

    // Enhanced drag and drop functionality
    setupDragAndDrop(uploadZone, fileInput);
    
    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            processFile(e.target.files[0]);
        }
    });

    // Upload zone click handler (but not when clicking the analyze button)
    uploadZone.addEventListener('click', (e) => {
        // Don't trigger file input if clicking on the analyze button
        if (e.target.id === 'analyzeButton' || e.target.closest('#analyzeButton')) {
            return;
        }
        // Don't trigger if already has a file and analyze button is visible
        if (document.getElementById('analyzeButton') && document.getElementById('analyzeButton').style.display !== 'none') {
            return;
        }
        fileInput.click();
    });

    function setupDragAndDrop(uploadZone, fileInput) {
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', (e) => {
            if (!uploadZone.contains(e.relatedTarget)) {
                uploadZone.classList.remove('dragover');
            }
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && validateFile(files[0])) {
                fileInput.files = files;
                processFile(files[0]);
            }
        });
    }

    function validateFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/dicom'];
        const maxSize = 50 * 1024 * 1024; // 50MB
        
        // Check file type
        const isValidType = validTypes.some(type => 
            file.type.includes(type.split('/')[1]) || 
            file.name.toLowerCase().includes(type.split('/')[1]) ||
            file.name.toLowerCase().includes('dcm') ||
            file.name.toLowerCase().includes('dicom')
        );
        
        if (!isValidType) {
            showNotification('Please upload a valid medical image file (JPEG, PNG, or DICOM)', 'error');
            return false;
        }
        
        if (file.size > maxSize) {
            showNotification('File size must be less than 50MB', 'error');
            return false;
        }
        
        return true;
    }

    function processFile(file) {
        if (!validateFile(file)) return;

        // Clear any existing analyze button first
        const existingAnalyzeBtn = document.getElementById('analyzeButton');
        if (existingAnalyzeBtn) {
            existingAnalyzeBtn.remove();
        }

        // Update file details
        updateFileDetails(file);
        
        // Convert to base64 and show preview
        convertToBase64AndPreview(file);
    }

    function convertToBase64AndPreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            currentImageBase64 = e.target.result;
            
            // Display preview immediately
            previewContainer.innerHTML = `
                <img src="${currentImageBase64}" alt="CT Scan Preview" class="preview-image">
                <div class="status-indicator"></div>
            `;
            
            // Update file status to show preview is ready
            updateFileStatus('Ready for Analysis');
            
            // Show analyze button or auto-submit (user preference)
            showAnalyzeButton();
        };
        
        reader.onerror = function() {
            showNotification('Error reading file. Please try again.', 'error');
            updateFileStatus('Error');
        };
        
        reader.readAsDataURL(file);
    }

    function showAnalyzeButton() {
        // Remove any existing analyze button first
        const existingBtn = document.getElementById('analyzeButton');
        if (existingBtn) {
            existingBtn.remove();
        }
        
        // Create new analyze button
        const analyzeBtn = document.createElement('button');
        analyzeBtn.id = 'analyzeButton';
        analyzeBtn.type = 'button';
        analyzeBtn.className = 'upload-button';
        analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze Image';
        analyzeBtn.style.marginTop = '16px';
        analyzeBtn.style.width = '100%';
        
        // Prevent event bubbling to parent
        analyzeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            submitAnalysisBase64();
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        });
        
        // Add button to upload zone
        const uploadZone = document.getElementById('uploadZone');
        uploadZone.appendChild(analyzeBtn);
    }

    function updateFileDetails(file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
        document.getElementById('fileType').textContent = file.type || 'Medical Image';
        document.getElementById('fileStatus').textContent = 'Processing';
        fileDetails.style.display = 'block';
    }

    function submitAnalysisBase64() {
        if (!currentImageBase64) {
            showNotification('No image data available. Please select a file first.', 'error');
            return;
        }

        loadingOverlay.style.display = 'flex';
        
        // Update status indicator to processing
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
            statusIndicator.classList.add('processing');
        }
        
        // Send base64 image to server
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImageBase64
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            loadingOverlay.style.display = 'none';
            
            if (data.success) {
                displayResults(data);
                updateFileStatus('Analysis Complete');
                showNotification('Analysis completed successfully!', 'success');
                
                // Replace analyze button with "Upload New Image" button
                const analyzeBtn = document.getElementById('analyzeButton');
                if (analyzeBtn) {
                    analyzeBtn.innerHTML = '<i class="fas fa-upload"></i> Upload New Image';
                    analyzeBtn.disabled = false;
                    analyzeBtn.onclick = (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        window.medicalInterface.resetInterface();
                    };
                }
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }
        })
        .catch(error => {
            console.error('Analysis error:', error);
            loadingOverlay.style.display = 'none';
            updateFileStatus('Analysis Failed');
            showNotification(`Analysis failed: ${error.message}`, 'error');
            
            // Re-enable analyze button on error
            const analyzeBtn = document.getElementById('analyzeButton');
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-redo"></i> Retry Analysis';
            }
        });
    }

    function displayResults(data) {
        // Remove existing results panel if any
        const existingResults = document.querySelector('.results-panel');
        if (existingResults) {
            existingResults.remove();
        }

        // Create new results panel
        const resultsPanel = document.createElement('div');
        resultsPanel.className = 'results-panel fade-in';
        
        // Determine if result is normal or abnormal
        const isNormal = data.prediction.toLowerCase() === 'normal';
        const resultClass = isNormal ? 'normal' : 'cancer';
        
        resultsPanel.innerHTML = `
            <div class="results-header">
                <div class="results-title">Analysis Complete</div>
                <div class="results-subtitle">AI-powered diagnostic assessment</div>
            </div>
            <div class="results-grid">
                <div class="result-item">
                    <div class="result-label">Diagnosis</div>
                    <div class="result-value ${resultClass}">
                        ${data.prediction.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div class="result-description">AI Classification</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Confidence</div>
                    <div class="result-confidence">${data.confidence_percentage}</div>
                    <div class="result-description">Model Certainty</div>
                </div>
            </div>
        `;

        // Insert after the main interface
        const mainInterface = document.querySelector('.main-interface');
        mainInterface.parentNode.insertBefore(resultsPanel, mainInterface.nextSibling);

        // Update status indicator
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
            statusIndicator.classList.remove('processing');
            statusIndicator.style.background = isNormal ? 'var(--success-green)' : 'var(--warning-amber)';
        }
    }

    function updateFileStatus(status) {
        const fileStatus = document.getElementById('fileStatus');
        if (fileStatus) {
            fileStatus.textContent = status;
        }
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());

        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    function getNotificationIcon(type) {
        const icons = {
            error: 'exclamation-circle',
            success: 'check-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }
}

// Export functions for global access if needed
window.medicalInterface = {
    showNotification: function(message, type) {
        const event = new CustomEvent('showNotification', {
            detail: { message, type }
        });
        document.dispatchEvent(event);
    },
    
    getCurrentImage: function() {
        return currentImageBase64;
    },
    
    resetInterface: function() {
        currentImageBase64 = null;
        const fileDetails = document.getElementById('fileDetails');
        const previewContainer = document.getElementById('previewContainer');
        const existingResults = document.querySelector('.results-panel');
        const analyzeBtn = document.getElementById('analyzeButton');
        const fileInput = document.getElementById('fileInput');
        
        if (fileDetails) fileDetails.style.display = 'none';
        if (existingResults) existingResults.remove();
        if (analyzeBtn) analyzeBtn.remove();
        if (fileInput) fileInput.value = '';
        
        if (previewContainer) {
            previewContainer.innerHTML = `
                <div class="preview-placeholder">
                    <i class="fas fa-image"></i>
                    <p>Preview will appear here</p>
                </div>
            `;
        }
        
        showNotification('Interface reset. Ready for new upload.', 'info');
    }
};