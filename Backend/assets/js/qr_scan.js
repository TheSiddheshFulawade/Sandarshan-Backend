document.addEventListener("DOMContentLoaded", function () {
  // Safe element selection with fallback
  function safeSelect(selector) {
    const element = document.querySelector(selector);
    if (!element) {
      console.warn(`Element not found: ${selector}`);
    }
    return element;
  }

  // Element Selections
  const videoElement = safeSelect("#videoElement");
  const captureCanvas = safeSelect("#captureCanvas");
  const startCameraBtn = safeSelect("#startCamera");
  const captureImageBtn = safeSelect("#captureImage");
  const qrResultSection = safeSelect("#qrResultContent");

  // Face Recognition Elements
  const faceVideoElement = safeSelect("#faceVideoElement");
  const faceCaptureCanvas = safeSelect("#faceCaptureCanvas");
  const faceResultSection = safeSelect("#faceResultContent");
  const faceRecognitionCard = safeSelect("#faceRecognitionCard");
  const countdownDisplay = safeSelect("#countdownDisplay");

  // Transition and Popup Elements
  const qrSection = safeSelect("#qrSection");
  const faceSection = safeSelect("#faceSection");
  const successPopup = safeSelect("#successPopup");
  const alertPopup = safeSelect("#alertPopup");

  // Error Message
  const errorMessage = safeSelect("#errorMessage");

  const criticalElements = [
    videoElement,
    captureCanvas,
    startCameraBtn,
    captureImageBtn,
    qrResultSection,
    faceVideoElement,
    faceCaptureCanvas,
    faceResultSection,
    faceRecognitionCard,
    countdownDisplay,
    qrSection,
    faceSection,
    successPopup,
    alertPopup,
  ];

  const missingElements = criticalElements.filter((el) => !el);
  if (missingElements.length > 0) {
    console.error("Missing critical elements:", missingElements);
    return; // Exit if critical elements are missing
  }

  let qrStream = null;
  let faceStream = null;
  let qrScanData = null;
  let faceDetectionInterval = null;
  let countdownTimer = null;
  let detectionContext = null;
  let alertShown = false;

  // Get CSRF token
  const csrfToken = document.querySelector("[name=csrfmiddlewaretoken]");

  // Start QR Camera Function
  if (startCameraBtn) {
    startCameraBtn.addEventListener("click", async function () {
      try {
        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("Camera access not supported");
        }

        qrStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });

        if (videoElement) {
          videoElement.srcObject = qrStream;

          // Wait for video to be ready
          videoElement.onloadedmetadata = () => {
            videoElement.play();
            if (startCameraBtn) startCameraBtn.disabled = true;
            if (captureImageBtn) captureImageBtn.disabled = false;
          };
        }

        if (errorMessage) {
          errorMessage.textContent = "";
        }

        if (qrResultSection) {
          qrResultSection.innerHTML = "";
        }
      } catch (error) {
        console.error("Camera error:", error);

        if (errorMessage) {
          // Provide more specific error messages
          let errorMsg = "Camera access denied or not available.";
          if (error.name === "NotAllowedError") {
            errorMsg =
              "Camera permission was denied. Please check your browser settings.";
          } else if (error.name === "NotFoundError") {
            errorMsg = "No camera found on this device.";
          }

          errorMessage.textContent = errorMsg;
        }

        if (startCameraBtn) startCameraBtn.disabled = false;
        if (captureImageBtn) captureImageBtn.disabled = true;
      }
    });
  }

  // Capture QR Image Function
  if (captureImageBtn) {
    captureImageBtn.addEventListener("click", function () {
      if (videoElement && captureCanvas) {
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;

        const context = captureCanvas.getContext("2d");
        context.drawImage(videoElement, 0, 0);

        const imageDataUrl = captureCanvas.toDataURL("image/png");
        processQRCode(imageDataUrl);
      }
    });
  }

  // Process QR Code
  function processQRCode(imageData) {
    if (!csrfToken) {
      console.error("CSRF token not found");
      return;
    }

    fetch("/process_qr_code/", {
      method: "POST",
      headers: {
        "X-CSRFToken": csrfToken.value,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: `image=${encodeURIComponent(imageData)}`,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        console.log(data.success);
        if (data.success) {
          qrScanData = data;

          // Update QR Result Section
          if (qrResultSection) {
            console.log(qrResultSection);
            qrResultSection.innerHTML = `
                    <p><strong>Name:</strong> ${data.name}</p>
                    <p><strong>Type:</strong> ${data.type}</p>
                    <p><strong>Email:</strong> ${data.email}</p>
                    <p><strong>Office Location:</strong> ${data.office_location}</p>
                `;
          }

          // Enable Face Recognition Card
          if (faceRecognitionCard) {
            faceRecognitionCard.style.display = "block";
          }

          // Show Success Popup and Transition
          showSuccessPopup("QR Code Verified!", 3000);

          setTimeout(() => {
            transitionToFaceRecognition();
          }, 3000);
        } else {
          // Handle unsuccessful QR code processing
          if (errorMessage) {
            errorMessage.textContent =
              data.message || "QR Code verification failed.";
          }

          if (qrResultSection) {
            qrResultSection.innerHTML = "";
          }
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        console.log(errorMessage);
        if (errorMessage) {
          errorMessage.textContent =
            "An error occurred while processing the QR code.";
        }
      });
  }

  // Start Continuous Face Recognition
  function startContinuousFaceRecognition() {
    navigator.mediaDevices
      .getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })
      .then((stream) => {
        faceStream = stream;
        faceVideoElement.srcObject = stream;

        // Setup canvas for drawing bounding boxes
        const canvas = document.createElement("canvas");
        canvas.width = faceVideoElement.videoWidth || 640;
        canvas.height = faceVideoElement.videoHeight || 480;
        detectionContext = canvas.getContext("2d");
        faceVideoElement.parentNode.insertBefore(
          canvas,
          faceVideoElement.nextSibling
        );
        // canvas.style.position = 'absolute';
        // canvas.style.top = faceVideoElement.offsetTop + 'px';
        // canvas.style.left = faceVideoElement.offsetLeft + 'px';

        // Start countdown and face detection
        startCountdownAndDetection();
      })
      .catch((error) => {
        console.error("Face camera error:", error);
        errorMessage.textContent = "Face camera access denied.";
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
      countdownDisplay.style.display = "block";
    } else {
      console.error("Countdown display element not found");
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
    if (typeof timeLeft === "undefined" || timeLeft <= 0) {
      stopFaceRecognition(false);
      return;
    }

    // Clear previous canvas drawings
    detectionContext.clearRect(
      0,
      0,
      detectionContext.canvas.width,
      detectionContext.canvas.height
    );

    // Capture current frame
    const canvas = document.createElement("canvas");
    canvas.width = faceVideoElement.videoWidth;
    canvas.height = faceVideoElement.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(faceVideoElement, 0, 0);

    const imageDataUrl = canvas.toDataURL("image/png");

    // Send for processing
    fetch("/process_detection/", {
      method: "POST",
      headers: {
        "X-CSRFToken": csrfToken.value,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: `image=${encodeURIComponent(imageDataUrl)}`,
    })
      .then((response) => response.json())
      .then((data) => {
        // Only process and draw if time is still remaining
        if (timeLeft > 0) {
          if (
            data.success &&
            (data.faces.length > 0 || data.objects.length > 0)
          ) {
            // Stop detection if faces or objects found
            stopFaceRecognition(true, data);
          }
        } else {
          // Clear the canvas if time has expired
          detectionContext.clearRect(
            0,
            0,
            detectionContext.canvas.width,
            detectionContext.canvas.height
          );
        }
      })
      .catch((error) => {
        console.error("Detection error:", error);
      });
  }

  async function triggerProxyAlert() {
    try {
      // Create an Audio object with the specific path
      const audioPath = "/static/audio/Alert.mp3";
      const alertSound = new Audio(audioPath);

      // Play the alert sound
      await new Promise((resolve, reject) => {
        alertSound.addEventListener("ended", resolve);
        alertSound.addEventListener("error", reject);
        alertSound.play().catch(reject);
      });

      // After sound finishes, speak the alert
      await new Promise((resolve) => {
        if ("speechSynthesis" in window) {
          //   const utterance = new SpeechSynthesisUtterance("Proxy Debard");

          // Optionally customize voice
          utterance.rate = 1;
          utterance.pitch = 1;

          const voices = window.speechSynthesis.getVoices();
          const englishVoice = voices.find(
            (voice) =>
              voice.lang.includes("en-") &&
              (voice.name.includes("Google") ||
                voice.name.includes("Microsoft"))
          );

          if (englishVoice) {
            utterance.voice = englishVoice;
          }

          utterance.onend = resolve;
          window.speechSynthesis.speak(utterance);
        } else {
          resolve();
        }
      });
    } catch (error) {
      console.error("Alert sequence error:", error);
    }
  }

  function playAlertSound() {
    try {
      // Create an Audio object with the specific path
      const audioPath = "/assets/audio/Alert.mp3";
      const alertSound = new Audio(audioPath);

      // Play the audio
      alertSound
        .play()
        .then(() => {
          console.log("Alert sound played successfully");
        })
        .catch((error) => {
          console.error("Error playing alert sound:", error);
        });
    } catch (error) {
      console.error("Failed to create audio object:", error);
    }

  }

  function speakAlert(message) {
    // Check if browser supports Web Speech API
    if ("speechSynthesis" in window) {
      // Create a new SpeechSynthesisUtterance object
      const utterance = new SpeechSynthesisUtterance(message);

      // Optional: Customize voice properties
      utterance.rate = 1; // Normal speech rate
      utterance.pitch = 1; // Normal pitch

      // Optionally, you can choose a specific voice
      const voices = window.speechSynthesis.getVoices();
      // Try to find an English voice
      const englishVoice = voices.find(
        (voice) =>
          voice.lang.includes("en-") &&
          (voice.name.includes("Google") || voice.name.includes("Microsoft"))
      );

      if (englishVoice) {
        utterance.voice = englishVoice;
      }

      // Speak the message
      window.speechSynthesis.speak(utterance);
    } else {
      console.warn("Text-to-speech not supported");
    }
  }
  // Stop Face Recognition
  function stopFaceRecognition(foundMatch, detectionData = null) {
    // Clear intervals
    if (countdownTimer) clearInterval(countdownTimer);
    if (faceDetectionInterval) clearInterval(faceDetectionInterval);
    if (qrStream) {
      qrStream.getTracks().forEach((track) => track.stop());
      videoElement.srcObject = null;
    }

    // Hide countdown
    countdownDisplay.style.display = "none";

    // Stop video stream
    if (faceStream) {
      faceStream.getTracks().forEach((track) => track.stop());
      faceVideoElement.srcObject = null;
    }

    if (detectionContext) {
      detectionContext.clearRect(
        0,
        0,
        detectionContext.canvas.width,
        detectionContext.canvas.height
      );
    }

    // Handle verification result
    if (qrScanData && detectionData.faces.length > 0) {
      const matchedFace = detectionData.faces.find(
        (face) =>
          face.name && face.name.toLowerCase() === qrScanData.name.toLowerCase()
      );

      if (!alertShown) {
        alertShown = true;

        if (matchedFace) {
          // Show success popup
          showSuccessPopup("Person Verified Successfully!", 3000);
        } else {
          // Show alert popup
          showAlertPopup("ALERT: Proxy Attempt Detected!", 3000);
        }
      }
    } else {
      // Show alert for no detection
      showAlertPopup("No face or object detected within the time limit.", 3000);
    }

    // Process results if a match was found
    if (foundMatch && detectionData) {
      let faceHtml = "";
      detectionData.faces.forEach((face) => {
        faceHtml += `
                    <p><strong>Recognized Name:</strong> ${
                      face.name || "Unknown"
                    }</p>
                    <p><strong>Confidence:</strong> ${(
                      face.confidence * 100
                    ).toFixed(2)}%</p>
                `;
      });

      let objectHtml = "";
      if (detectionData.objects.length > 0) {
        objectHtml += "<p><strong>Detected Objects:</strong></p>";
        detectionData.objects.forEach((obj) => {
          objectHtml += `
                        <p>${obj.class_name} (${(obj.confidence * 100).toFixed(
            2
          )}%)</p>
                    `;
        });
      }

      faceResultSection.innerHTML = faceHtml + objectHtml;

      function showSuccessPopup(
        message = "Verified Successfully!",
        duration = 4000
      ) {
        const successPopup = document.getElementById("successPopup");
        if (!successPopup) {
          console.error("Success popup element not found!");
          return;
        }
        const successMessage = successPopup.querySelector("#successMessage");
        if (successMessage) {
          successMessage.textContent = message;
        }
        successPopup.style.display = "flex"; // Ensure popup is visible

        // Automatically hide popup after duration
        setTimeout(() => {
          successPopup.style.display = "none";
          resetToQRScanner(); // Reset back to QR scanner
        }, duration);
      }

      function showAlertPopup(
        message = "Proxy Attempt Detected!",
        duration = 8000
      ) {
        const alertPopup = document.getElementById("alertPopup");
        if (!alertPopup) {
          console.error("Alert popup element not found!");
          return;
        }
        const alertMessage = alertPopup.querySelector("#alertMessage");
        if (alertMessage) {
          alertMessage.textContent = message;
        }
        alertPopup.style.display = "flex"; // Ensure popup is visible

        // Automatically hide popup after duration
        setTimeout(() => {
          alertPopup.style.display = "none";
          resetToQRScanner(); // Reset back to QR scanner
        }, duration);
      }

      function transitionToFaceRecognition() {
        if (!qrSection || !faceSection) {
          console.warn("QR or Face section not found");
          return;
        }

        // Slide out QR section
        qrSection.classList.add("slide-out-left");

        setTimeout(() => {
          qrSection.style.display = "none";
          faceSection.style.display = "block";
          faceSection.classList.add("slide-in-right");

          // Start face recognition immediately
          startContinuousFaceRecognition();
        }, 500);
      }

      function resetToQRScanner() {
        if (!qrSection || !faceSection) return;
    
        // Transition animations
        faceSection.classList.remove("slide-in-right");
        faceSection.classList.add("slide-out-left");
    
        setTimeout(() => {
            faceSection.style.display = "none";
            qrSection.style.display = "block";
            qrSection.classList.add("slide-in-right");
        }, 500);
    
        // Stop QR scanner stream
        if (qrStream) {
            qrStream.getTracks().forEach((track) => track.stop());
            qrStream = null; // Clear reference to avoid reusing the stream
            videoElement.srcObject = null; // Clear the video element
        }
    
        // Stop face recognition stream
        if (faceStream) {
            faceStream.getTracks().forEach((track) => track.stop());
            faceStream = null; // Clear reference to avoid reusing the stream
            faceVideoElement.srcObject = null; // Clear the face video element
        }
    
        // Reset variables and UI
        qrScanData = null;
        alertShown = false;
        detectionData = null;
        timeLeft = 10;
    
        // Clear the result sections
        const faceResultSection = document.getElementById("faceResultSection");
        if (faceResultSection) faceResultSection.innerHTML = "";
    
        const qrResultSection = document.getElementById("qrResultSection");
        if (qrResultSection) qrResultSection.innerHTML = "<p>QR Scan Result:</p>";
    
        const countdownDisplay = document.getElementById("countdownDisplay");
        if (countdownDisplay) {
            countdownDisplay.textContent = "";
            countdownDisplay.style.display = "none";
        }
    
        const successPopup = document.getElementById("successPopup");
        const alertPopup = document.getElementById("alertPopup");
        if (successPopup) successPopup.style.display = "none";
        if (alertPopup) alertPopup.style.display = "none";
    
        // Reset button states
        const startCameraButton = document.getElementById("startCamera");
        const captureImageButton = document.getElementById("captureImage");
        if (startCameraButton) {
            startCameraButton.disabled = false; // Re-enable "Start Camera" button
        }
        if (captureImageButton) {
            captureImageButton.disabled = true; // Disable "Capture QR" button
        }
    
        console.log("Reset complete. Ready for a new person.");
    }
    
    
    document.getElementById("startCamera").addEventListener("click", () => {
        startQRScanner();
    
        // Update button states
        const startCameraButton = document.getElementById("startCamera");
        const captureQRButton = document.getElementById("captureImage");
    
        if (startCameraButton) {
            startCameraButton.disabled = true; // Disable "Start Camera" button
        }
    
        if (captureQRButton) {
            captureQRButton.disabled = false; // Enable "Capture QR" button
        }
    });
    
    
    
      
      function startQRScanner() {
        if (!qrSection || !videoElement) {
            console.error("QR scanner elements not found!");
            return;
        }
    
        navigator.mediaDevices
            .getUserMedia({ video: true })
            .then((stream) => {
                qrStream = stream; // Assign stream for resetting later
                videoElement.srcObject = stream;
                videoElement.play();
    
                console.log("QR scanner started successfully.");
            })
            .catch((err) => {
                console.error("Error starting QR scanner:", err);
            });
    }
    

      let alertShown = false;

      // Modify the verification code in stopFaceRecognition function:
      if (qrScanData && detectionData.faces.length > 0) {
        const matchedFace = detectionData.faces.find(
          (face) =>
            face.name &&
            face.name.toLowerCase() === qrScanData.name.toLowerCase()
        );

        if (!alertShown) {
          alertShown = true;
          if (matchedFace) {
            showSuccessPopup("Person Verified Successfully!");
            // Reset flag after a delay if needed
            setTimeout(() => {
              alertShown = false;
            }, 3000);
          } else {
            showAlertPopup(
              "ALERT: Person Verification Failed! Possible Proxy Attempt."
            );
            playAlertSound();
            speakAlert("Proxy Debard");
            triggerProxyAlert();
          }
        }
      }
    } else {
      // No match found within 10 seconds
      alert("No face or object detected within the time limit.");
      faceResultSection.innerHTML = "No detection found.";
    }
  }
  window.showSuccessPopup = function (message, duration) {
    const successPopup = document.getElementById("successPopup");
    const successMessage = document.getElementById("successMessage");

    if (!successPopup) {
        console.error("Success popup element not found!");
        return;
    }

    // Set the popup message
    if (successMessage) {
        successMessage.textContent = message;
    }

    // Apply inline styles to center the popup
    successPopup.style.position = "fixed";
    successPopup.style.top = "50%";
    successPopup.style.left = "50%";
    successPopup.style.transform = "translate(-50%, -50%)";
    successPopup.style.zIndex = "1000";
    successPopup.style.backgroundColor = "#fff";
    successPopup.style.padding = "20px";
    successPopup.style.borderRadius = "8px";
    successPopup.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.2)";
    successPopup.style.textAlign = "center";
    successPopup.style.display = "flex";
    successPopup.style.flexDirection = "column";
    successPopup.style.alignItems = "center";
    successPopup.style.justifyContent = "center";

    // Show the popup and hide it after the specified duration
    setTimeout(() => {
        successPopup.style.display = "none";
    }, duration);
};


  window.transitionToFaceRecognition = function () {
    const qrSection = document.getElementById("qrSection");
    const faceSection = document.getElementById("faceSection");

    if (!qrSection || !faceSection) {
      console.warn("QR or Face section not found");
      return;
    }

    // Slide out QR section
    qrSection.classList.add("slide-out-left");

    setTimeout(() => {
      qrSection.style.display = "none";
      faceSection.style.display = "block";
      faceSection.classList.add("slide-in-right");

      // Start face recognition immediately
      startContinuousFaceRecognition();
    }, 500);
  };

  window.showAlertPopup = showAlertPopup;
});
