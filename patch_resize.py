import os
from PIL import Image

def find_small_tif_images(folder_path, min_size=28):
    small_images = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".tif"):
            filepath = os.path.join(folder_path, filename)
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    if width < min_size or height < min_size:
                        small_images.append((filename, width, height))
            except Exception as e:
                print(f"Failed to open {filename}: {e}")

    return small_images


def resize_image_if_too_small(image_path, min_size=28):
    with Image.open(image_path) as img:
        width, height = img.size
        if width >= min_size and height >= min_size:
            return None

        new_width = max(width, min_size)
        new_height = max(height, min_size)
        resized_img = img.resize((new_width, new_height), Image.BICUBIC)

        resized_img.save(image_path)

        return (os.path.basename(image_path), width, height, new_width, new_height)

folder = "data/patch/"
results = find_small_tif_images(folder)

if results:
    print("Images smaller than 28x28:")
    for name, w, h in results:
        print(f"{name}: {w}x{h}")
        image_path = os.path.join(folder, name)
        resize_result = resize_image_if_too_small(image_path)
else:
    print("No small images found.")

