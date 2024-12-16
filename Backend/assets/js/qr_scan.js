document.addEventListener('DOMContentLoaded', function() {
    // QR Scan Elements
    const videoElement = document.getElementById('videoElement');
    const captureCanvas = document.getElementById('captureCanvas');
    const startCameraBtn = document.getElementById('startCamera');
    const captureImageBtn = document.getElementById('captureImage');
    const qrResultSection = document.getElementById('qrResultContent');
    
    // Face Recognition Elements
    const faceVideoElement = document.getElementById('faceVideoElement');
    const faceCaptureCanvas = document.getElementById('faceCaptureCanvas');
    const faceResultSection = document.getElementById('faceResultContent');
    const faceRecognitionCard = document.getElementById('faceRecognitionCard');
    const countdownDisplay = document.getElementById('countdownDisplay');
    
    // Error Message
    const errorMessage = document.getElementById('errorMessage');

    let qrStream = null;
    let faceStream = null;
    let qrScanData = null;
    let faceDetectionInterval = null;
    let countdownTimer = null;
    let detectionContext = null;

    // Get CSRF token
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]');

    // Start QR Camera Function
    startCameraBtn.addEventListener('click', async function() {
        try {
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera access not supported');
            }

            qrStream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });

            videoElement.srcObject = qrStream;
            
            // Wait for video to be ready
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                startCameraBtn.disabled = true;
                captureImageBtn.disabled = false;
            };

            errorMessage.textContent = '';
            qrResultSection.innerHTML = '';
        } catch (error) {
            console.error('Camera error:', error);
            
            // Provide more specific error messages
            let errorMsg = 'Camera access denied or not available.';
            if (error.name === 'NotAllowedError') {
                errorMsg = 'Camera permission was denied. Please check your browser settings.';
            } else if (error.name === 'NotFoundError') {
                errorMsg = 'No camera found on this device.';
            }
            
            errorMessage.textContent = errorMsg;
            startCameraBtn.disabled = false;
            captureImageBtn.disabled = true;
        }
    });

    // Capture QR Image Function
    captureImageBtn.addEventListener('click', function() {
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;

        const context = captureCanvas.getContext('2d');
        context.drawImage(videoElement, 0, 0);

        const imageDataUrl = captureCanvas.toDataURL('image/png');
        processQRCode(imageDataUrl);
    });

    // Process QR Code
    function processQRCode(imageData) {
        if (!csrfToken) {
            console.error('CSRF token not found');
            return;
        }

        fetch('/process_qr_code/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrfToken.value,
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `image=${encodeURIComponent(imageData)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                qrScanData = data;
                qrResultSection.innerHTML = `
                    <p><strong>Name:</strong> ${data.name}</p>
                    <p><strong>Type:</strong> ${data.type}</p>
                    <p><strong>Email:</strong> ${data.email}</p>
                    <p><strong>Office Location:</strong> ${data.office_location}</p>
                `;
                
                // Enable Face Recognition Card
                faceRecognitionCard.style.display = 'block';
                startContinuousFaceRecognition();
            } else {
                errorMessage.textContent = data.message;
                qrResultSection.innerHTML = '';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            errorMessage.textContent = 'An error occurred while processing the QR code.';
        });
    }

    // Start Continuous Face Recognition
    function startContinuousFaceRecognition() {
        navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        })
        .then(stream => {
            faceStream = stream;
            faceVideoElement.srcObject = stream;

            // Setup canvas for drawing bounding boxes
            const canvas = document.createElement('canvas');
            canvas.width = faceVideoElement.videoWidth || 640;
            canvas.height = faceVideoElement.videoHeight || 480;
            detectionContext = canvas.getContext('2d');
            faceVideoElement.parentNode.insertBefore(canvas, faceVideoElement.nextSibling);
            canvas.style.position = 'absolute';
            canvas.style.top = faceVideoElement.offsetTop + 'px';
            canvas.style.left = faceVideoElement.offsetLeft + 'px';

            // Start countdown and face detection
            startCountdownAndDetection();
        })
        .catch(error => {
            console.error('Face camera error:', error);
            errorMessage.textContent = 'Face camera access denied.';
        });
    }

    let timeLeft = 10;

// Start Countdown and Face Detection
function startCountdownAndDetection() {
    // Reset timeLeft to 10
    timeLeft = 10;

    // Ensure countdownDisplay exists before using it
    if (countdownDisplay) {
        countdownDisplay.textContent = `Time Remaining: ${timeLeft} seconds`;
        countdownDisplay.style.display = 'block';
    } else {
        console.error('Countdown display element not found');
        return;
    }

    // Countdown timer
    countdownTimer = setInterval(() => {
        timeLeft--;
        
        if (countdownDisplay) {
            countdownDisplay.textContent = `Time Remaining: ${timeLeft} seconds`;
        }

        if (timeLeft <= 0) {
            stopFaceRecognition(false);
        }
    }, 1000);

    // Face detection interval
    faceDetectionInterval = setInterval(() => {
        captureAndProcessFace();
    }, 500); // Process every 500ms
}

    function captureAndProcessFace() {
        // Check if time has expired

        if (typeof timeLeft === 'undefined' || timeLeft <= 0) {
            stopFaceRecognition(false);
            return;
        }
    
        // Clear previous canvas drawings
        detectionContext.clearRect(0, 0, detectionContext.canvas.width, detectionContext.canvas.height);
    
        // Capture current frame
        const canvas = document.createElement('canvas');
        canvas.width = faceVideoElement.videoWidth;
        canvas.height = faceVideoElement.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(faceVideoElement, 0, 0);
    
        const imageDataUrl = canvas.toDataURL('image/png');
        
        // Send for processing
        fetch('/process_detection/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrfToken.value,
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `image=${encodeURIComponent(imageDataUrl)}`
        })
        .then(response => response.json())
        .then(data => {
            // Only process and draw if time is still remaining
            if (timeLeft > 0) {
                if (data.success && (data.faces.length > 0 || data.objects.length > 0)) {
                    // Stop detection if faces or objects found
                    stopFaceRecognition(true, data);
                }
    
                // Draw bounding boxes for faces
                data.faces.forEach(face => {
                    const [x1, y1, x2, y2] = face.bbox;
                    detectionContext.beginPath();
                    detectionContext.rect(x1, y1, x2 - x1, y2 - y1);
                    detectionContext.lineWidth = 2;
                    detectionContext.strokeStyle = 'green';
                    detectionContext.stroke();
    
                    // Add name label
                    detectionContext.font = '14px Arial';
                    detectionContext.fillStyle = 'green';
                    detectionContext.fillText(
                        face.name || 'Unknown', 
                        x1, 
                        y1 > 20 ? y1 - 10 : y1 + 20
                    );
                });
    
                // Draw bounding boxes for objects
                data.objects.forEach(obj => {
                    const [x1, y1, x2, y2] = obj.bbox;
                    detectionContext.beginPath();
                    detectionContext.rect(x1, y1, x2 - x1, y2 - y1);
                    detectionContext.lineWidth = 2;
                    detectionContext.strokeStyle = 'red';
                    detectionContext.stroke();
    
                    // Add object label
                    detectionContext.font = '14px Arial';
                    detectionContext.fillStyle = 'red';
                    detectionContext.fillText(
                        `${obj.class_name} (${(obj.confidence * 100).toFixed(2)}%)`, 
                        x1, 
                        y1 > 20 ? y1 - 10 : y1 + 20
                    );
                });
            }
        })
        .catch(error => {
            console.error('Detection error:', error);
        });
    }

    // Stop Face Recognition
    function stopFaceRecognition(foundMatch, detectionData = null) {
        // Clear intervals
        if (countdownTimer) clearInterval(countdownTimer);
        if (faceDetectionInterval) clearInterval(faceDetectionInterval);

        // Hide countdown
        countdownDisplay.style.display = 'none';

        // Stop video stream
        if (faceStream) {
            faceStream.getTracks().forEach(track => track.stop());
            faceVideoElement.srcObject = null;
        }

        // Process results if a match was found
        if (foundMatch && detectionData) {
            let faceHtml = '';
            detectionData.faces.forEach(face => {
                faceHtml += `
                    <p><strong>Recognized Name:</strong> ${face.name || 'Unknown'}</p>
                    <p><strong>Confidence:</strong> ${(face.confidence * 100).toFixed(2)}%</p>
                `;
            });

            let objectHtml = '';
            if (detectionData.objects.length > 0) {
                objectHtml += '<p><strong>Detected Objects:</strong></p>';
                detectionData.objects.forEach(obj => {
                    objectHtml += `
                        <p>${obj.class_name} (${(obj.confidence * 100).toFixed(2)}%)</p>
                    `;
                });
            }

            faceResultSection.innerHTML = faceHtml + objectHtml;

            // Verify name match if applicable
            if (qrScanData && detectionData.faces.length > 0) {
                const matchedFace = detectionData.faces.find(face => 
                    face.name && face.name.toLowerCase() === qrScanData.name.toLowerCase()
                );

                if (matchedFace) {
                    alert('Person Verified Successfully!');
                } else {
                    alert('ALERT: Person Verification Failed! Possible Proxy Attempt.');
                }
            }
        } else {
            // No match found within 10 seconds
            alert('No face or object detected within the time limit.');
            faceResultSection.innerHTML = 'No detection found.';
        }
    }
});