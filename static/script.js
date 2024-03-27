document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('image-input');
    const predictButton = document.getElementById('predict-button');
    const selectedFileText = document.getElementById('selected-file');
    const startWebcamButton = document.getElementById('start-webcam-button');

    imageInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            selectedFileText.textContent = 'Selected File: ' + file.name;
        } else {
            selectedFileText.textContent = '';
        }
    });

    predictButton.addEventListener('click', function(event) {
        event.preventDefault(); // Prevent form submission

        const file = imageInput.files[0];
        if (file) {
            try {
                const imageData = new FormData();
                imageData.append('image', file);
                sendToFlask(imageData);
            } catch (error) {
                console.error('Error reading file:', error);
            }
        }
    });

    

    startWebcamButton.addEventListener('click', function() {
        const webcamWindow = window.open('', 'Webcam Feed', 'width=640,height=480');
        
        if (!webcamWindow) {
            console.error('Failed to open webcam window.');
            return;
        }

        const webcamHTML = `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Webcam Feed</title>
                <style>
                    body, html {
                        margin: 0;
                        padding: 0;
                        width: 100%;
                        height: 100%;
                        overflow: hidden;
                        position: relative;
                    }
                    video {
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                    }
                    #webcam-emotion-prediction-result {
                        position: absolute;
                        top: 5%;
                        left: 20%;
                        transform: translate(-50%, -50%);
                        color: White;
                        z-index: 9999;
                        text-align: center;
                        font-weight: bold;
                        font-size: 24px;
                    
                    }
                </style>
            </head>
            <body>
                <video id="webcam-video" autoplay></video>
                <p id="webcam-emotion-prediction-result">Waiting for prediction...</p>
                <script>
                    navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(stream) {
                        const video = document.getElementById('webcam-video');
                        video.srcObject = stream;

                        setInterval(function() {
                            const canvas = document.createElement('canvas');
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            const context = canvas.getContext('2d');
                            context.drawImage(video, 0, 0, canvas.width, canvas.height);
                            canvas.toBlob(function(blob) {
                                sendWebcamFrameToFlask(blob);
                            }, 'image/jpeg');
                        }, 1000);
                    })
                    .catch(function(error) {
                        console.error('Error accessing webcam:', error);
                    });

                    async function sendWebcamFrameToFlask(blob) {
                        try {
                            const formData = new FormData();
                            formData.append('frame', blob, 'webcam_frame.jpg');
                            const response = await fetch('/webcam-feed', {
                                method: 'POST',
                                body: formData
                            });
                            const data = await response.json();
                            console.log('Webcam Prediction:', data.emotion);

                            const webcamPredictionResultElement = document.getElementById('webcam-emotion-prediction-result');
                            webcamPredictionResultElement.textContent = data.emotion;
                        } catch (error) {
                            console.error('Error sending webcam frame to Flask:', error);
                        }
                    }
                </script>
            </body>
            </html>
        `;

        webcamWindow.document.write(webcamHTML);
        
        window.addEventListener('keydown', function(event) {
            if (event.key === 'q') {
                webcamWindow.close();
            }
        });
    });
        
    
    
    
    

    async function sendToFlask(imageData) {
        try {
            console.log('Sending image data to Flask:', imageData);
            const response = await fetch('/predict', {
                method: 'POST',
                body: imageData
            });
            const data = await response.json();
            console.log('Prediction:', data.emotion);
            // Display the prediction result on the page
            displayPrediction(data.emotion);
        } catch (error) {
            console.error('Error sending image to Flask:', error);
        }
    }

    function displayPrediction(emotion) {
        // Update the HTML element with the prediction result
        const predictionResult = document.getElementById('prediction-result');
        predictionResult.textContent = 'Predicted Emotion: ' + emotion;
    }
});
    