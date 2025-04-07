from app import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Caption generation logic
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ""
    img_url = ""

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', caption="No file uploaded.")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', caption="No file selected.")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            caption = generate_caption(filepath)
            img_url = filepath

    return render_template('index.html', caption=caption, img_url=img_url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
