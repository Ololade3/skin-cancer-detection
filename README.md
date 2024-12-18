# Skin Cancer Detection Web App

This project is a web-based application for detecting skin cancer from uploaded images. It uses a pre-trained PyTorch model to classify images into categories, providing predictions and confidence scores. Below are the steps to set up, run, and deploy the application.

# Features

- Upload skin images for classification.

- Predict whether the uploaded image is cancerous or non-cancerous.

- Display the prediction result along with confidence percentage.

# Requirements

Ensure the following dependencies are installed on your system:

- Python (3.8 or above)

- PyTorch

- Flask

- torchvision

- PIL (Pillow)

Other required packages are listed in requirements.txt.

# Steps to Set Up and Run Locally

# 1. Clone the Repository

# Clone the repository to your local machine
https://github.com/Ololade3/skin-cancer-detection.git

# Navigate into the project directory
cd skin-cancer-detection

# 2. Install Dependencies

Ensure you have a virtual environment to manage dependencies

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# 3. Add the Model File

Ensure the model_full.pth file is placed in the model/ directory. If the directory does not exist, create it:

mkdir model

# Move the model_full.pth file into the model directory

# 4. Run the Application

Start the Flask application:

python app.py

# 5. Test the Application

Upload a skin image for detection.

View the prediction results.

# Project file Structure

skin-cancer-detection/
|-- app.py              # Main Flask application
|-- requirements.txt    # Dependencies for the project
|-- model/              # Folder for the pre-trained PyTorch model
|   -- model_full.pth  # The PyTorch model file
|-- templates/          # HTML templates
|   |-- index.html      # Main upload page
|   -- result.html     # Result display page


# Initialize git and add the remote repository
- git init
- git add .
- git commit -m "Initial commit"
- git branch -M main
- git remote add origin https://github.com/Ololade3/skin-cancer-detection.git
- git push -u origin main

License

This project is licensed under the MIT License.
