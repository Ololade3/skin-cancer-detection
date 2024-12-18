import os
from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

#  Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "model/model_full.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

#  image preprocessing pipeline
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    return transform(image).unsqueeze(0) 


@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', error="No file part in the request")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No selected file")

    try:
        # Open the image file
        image = Image.open(file).convert('RGB')
        
        # Preprocess the image
        input_tensor = preprocess_image(image).to(device)
        
        # model evaluation
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output[0], dim=0) 
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item() * 100
        
        # Map predicted_class to your class names
        class_names = ['Non-Cancerous', 'Cancerous']  
        prediction = class_names[predicted_class]
        
        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction, confidence=f"{confidence:.2f}%")
    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
