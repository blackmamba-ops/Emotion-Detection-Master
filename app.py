from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import cv2
import numpy as np
import logging
import google.generativeai as genai
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

app = Flask(__name__)


# Load the trained model
model = load_model("src\model.h5")  

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the trained image captioning model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Function to predict emotion from the image
def predict_emotion(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            resized_image = cv2.resize(roi_gray, (48, 48))
            resized_image = np.expand_dims(resized_image, axis=-1)
            resized_image = np.expand_dims(resized_image, axis=0)
            prediction = model.predict(resized_image)
            maxindex = int(np.argmax(prediction))
            emotion = emotion_dict[maxindex]
            return emotion
        
        return "No face detected"  # Return if no face is detected
    except Exception as e:
        logging.error('Error during emotion prediction: %s', e)
        return "Error during emotion prediction"
    

# Function to predict caption from the image
def predict_caption(image):
    try:
        pixel_values = caption_feature_extractor(images=image, return_tensors="pt").pixel_values
        output_ids = caption_model.generate(pixel_values)
        preds = caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]
    except Exception as e:
        logging.error('Error during caption prediction: %s', e)
        return "Error during caption prediction"
    

# Route to render the HTML page with the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info('Received POST request to /predict endpoint')

        # Get image data from the form submission
        if 'image' in request.files:
            image_data = request.files['image']
            logging.debug('Received image data')

            # Convert image data to numpy array
            nparr = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Log the shape of the image data for debugging
            logging.debug('Shape of image: %s', image.shape)

            # Predict emotion
            emotion = predict_emotion(image)
            logging.debug('Predicted emotion: %s', emotion)

            # Return JSON response
            return jsonify({'emotion': emotion})
        else:
            return jsonify({'error': 'No image data received'}), 400
    
    except Exception as e:
        logging.error('An error occurred: %s', e)
        return jsonify({'error': 'An error occurred'}), 500

# Route to handle webcam frames
@app.route('/webcam-feed', methods=['POST'])
def webcam_feed():
    try:
        logging.info('Received webcam frame')

        # Get image data from the form submission
        if 'frame' in request.files:
            frame_data = request.files['frame']
            logging.debug('Received frame data')

            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data.read(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Log the shape of the frame data for debugging
            logging.debug('Shape of frame: %s', frame.shape)

            # Predict emotion
            emotion = predict_emotion(frame)
            logging.debug('Predicted emotion: %s', emotion)

            # Return JSON response
            return jsonify({'emotion': emotion})
        else:
            return jsonify({'error': 'No frame data received'}), 400
    
    except Exception as e:
        logging.error('An error occurred: %s', e)
        return jsonify({'error': 'An error occurred'}), 500
    
# Route to handle webcam frames for image captioning
@app.route('/webcam-feed-caption', methods=['POST'])
def webcam_feed_caption():
    try:
        logging.info('Received webcam frame for caption prediction')

        # Get image data from the form submission
        if 'frame' in request.files:
            frame_data = request.files['frame']
            logging.debug('Received frame data for caption prediction')

            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data.read(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Predict caption
            caption = predict_caption(frame)
            logging.debug('Predicted caption: %s', caption)

            # Return JSON response
            return jsonify({'caption': caption})
        else:
            return jsonify({'error': 'No frame data received for caption prediction'}), 400
    
    except Exception as e:
        logging.error('An error occurred during caption prediction: %s', e)
        return jsonify({'error': 'An error occurred during caption prediction'}), 500


@app.route('/image-predict', methods=['POST'])
def image_predict():
    try:
        logging.info('Received POST request to /image-predict endpoint')

        # Get image data from the form submission
        if 'image' in request.files:
            image_data = request.files['image']
            logging.debug('Received image data')

            # Convert image data to numpy array
            nparr = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Log the shape of the image data for debugging
            logging.debug('Shape of image: %s', image.shape)

            # Predict caption
            caption = predict_caption(image)
            logging.debug('Predicted caption: %s', caption)

            # Return JSON response
            return jsonify({'caption': caption})
        else:
            return jsonify({'error': 'No image data received'}), 400
    
    except Exception as e:
        logging.error('An error occurred during image prediction: %s', e)
        return jsonify({'error': 'An error occurred during image prediction'}), 500
    






# Function to start conversation with GenerativeAI
def start_conversation(user_input):
    try:
        # Configure GenerativeAI API key
        genai.configure(api_key="YOUR API HERE")

        # Set up the model configuration
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        # Initialize GenerativeModel
        model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        # Start conversation
        convo = model.start_chat(history=[])

        # Send user input and get response
        convo.send_message(user_input)
        response = convo.last.text

        return response
    
    except Exception as e:
        logging.error('An error occurred during conversation: %s', e)
        return "An error occurred during conversation"

@app.route('/start_conversation', methods=['POST'])
def handle_start_conversation():
    try:
        user_input = request.json.get('user_input')

        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400

        response = start_conversation(user_input)

        return jsonify({'response': response})
    
    except Exception as e:
        logging.error('An error occurred: %s', e)
        return jsonify({'error': 'An error occurred'}), 500





if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
