# caller.py

from PIL import Image
import numpy as np
from image_processor import fix_image
from strategies import Strategies

def main():
    # Load image
    img = Image.open("medieval-landscape.png")
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_data = np.array(img)

    # Process with AVERAGE strategy
    processed, h, w = fix_image(
        img_data, 
        img.height, 
        img.width, 
        out_pix_width=4, 
        out_pix_height=4,
        strategy=Strategies.AVERAGE
    )

    # Convert to proper image format before saving
    if processed.dtype != np.uint8:
        processed = processed.astype(np.uint8)
    
    # Ensure proper shape (height, width, channels)
    if processed.ndim == 2:  # If grayscale
        processed = np.expand_dims(processed, axis=-1)
        processed = np.repeat(processed, 3, axis=-1)  # Convert to RGB
        if img.mode == 'RGBA':
            alpha = np.full(processed.shape[:-1] + (1,), 255, dtype=np.uint8)
            processed = np.concatenate([processed, alpha], axis=-1)
    
    # Save result
    Image.fromarray(processed).save("output.png")

if __name__ == "__main__":
    main()