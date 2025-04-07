# Ensure all libraries are installed
# pip install transformers torch pillow

from transformers import BlipProcessor, BlipForConditionalGeneration
from tkinter import Tk, filedialog, Label, Button
from PIL import Image, ImageTk
import torch

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate caption from image path
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Upload function to get image from user and display results
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        caption = generate_caption(file_path)
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        image_label.configure(image=img_tk)
        image_label.image = img_tk
        caption_label.config(text=f"Caption: {caption}")

# TKinter GUI setup
root = Tk()
root.title("🖼️ Image Caption Generator")
root.geometry("400x550")

Label(root, text="Upload an Image to Generate a Caption", font=("Arial", 14), wraplength=380).pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

Button(root, text="Upload Image", command=upload_image, font=("Arial", 12)).pack(pady=10)

caption_label = Label(root, text="", wraplength=350, font=("Arial", 12))
caption_label.pack(pady=20)

root.mainloop()
