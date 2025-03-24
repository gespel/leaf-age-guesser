from PIL import Image
import os

input_folder = "source_images"
output_folder = "destination_images"

size = (240, 240)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        img_resized.save(os.path.join(output_folder, filename))
        print("Converted {} to {}".format(filename, img_path))
