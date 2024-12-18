from flask import Flask, render_template
import os

app = Flask(__name__)

# Path to the folder containing processed images
PROCESSED_IMAGES_FOLDER = r'C:\me\DipProject\static\ProcessedImages'

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/results')
def home():
    # List all images in the processed images folder
    images = [img for img in os.listdir(PROCESSED_IMAGES_FOLDER) if img.endswith(('.png', '.jpg', '.jpeg'))]
    print(images)
    return render_template('results.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)