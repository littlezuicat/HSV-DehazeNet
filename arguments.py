import os
from PIL import Image

def center_crop(image, new_width, new_height):
    width, height = image.size
    left = (width - new_width) / 2 
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))

def crop_images_in_directory(directory, new_width, new_height):
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                cropped_img = center_crop(img, new_width, new_height)
                cropped_img.save(filepath)
                print(f'Cropped and saved: {filename}')

if __name__ == "__main__":
    directory = r'D:\Desktop\renyi\Dataset\SOTS\indoor\GT'
    new_width = 616  # Set your desired width
    new_height = 456  # Set your desired height
    crop_images_in_directory(directory, new_width, new_height)