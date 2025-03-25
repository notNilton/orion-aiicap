# image_manipulation.py

import numpy as np
from PIL import Image
import colorsys
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import warnings

def floyd_steinberg(image, levels=10):
    """
    Aplica Floyd-Steinberg em imagens RGB com paleta reduzida.
    """
    # Converte a imagem para um array NumPy
    pixels = np.array(image, dtype=float)
    width, height = image.size
    
    # Conta a quantidade de pixels na imagem original
    pixel_count = width * height
    print(f"Quantidade de pixels na imagem original: {pixel_count}")
    
    # Conta a quantidade de cores na imagem original
    color_count = count_colors(image)
    print(f"Quantidade de cores na imagem original: {color_count}")
    
    # Calcula o passo de quantização
    step = 255.0 / (levels - 1)
    
    # Itera sobre cada pixel
    for y in range(height - 1):  # Evita ultrapassar os limites
        for x in range(1, width - 1):  # Evita ultrapassar os limites
            # Processa cada canal de cor (R, G, B)
            for c in range(3):
                old_pixel = pixels[y, x, c]
                
                # Quantiza o pixel para o nível mais próximo
                new_pixel = round(old_pixel / step) * step
                new_pixel = np.clip(new_pixel, 0, 255)
                
                # Define o novo valor do pixel
                pixels[y, x, c] = new_pixel
                
                # Calcula o erro de quantização
                error = old_pixel - new_pixel
                
                # Distribui o erro para os vizinhos
                pixels[y, x + 1, c] += error * 7 / 16  # Direita
                pixels[y + 1, x - 1, c] += error * 3 / 16  # Inferior esquerdo
                pixels[y + 1, x, c] += error * 5 / 16  # Inferior
                pixels[y + 1, x + 1, c] += error * 1 / 16  # Inferior direito
    
    # Garante que os valores dos pixels estejam no intervalo [0, 255]
    pixels = np.clip(pixels, 0, 255)
    
    # Converte o array de volta para uma imagem PIL
    return Image.fromarray(pixels.astype(np.uint8), mode="RGB")

def pixelate_image(image, pixel_size=256):
    """
    Pixelates the image by reducing it to the specified pixel dimensions 
    and scaling it back up using nearest neighbor interpolation.
    
    Args:
        image (PIL.Image): Original image
        pixel_size (int): The desired pixel block size (e.g., 64 = 64x64 pixels)
                         Common values: 256, 128, 64, 32, 16
    
    Returns:
        PIL.Image: Pixelated image with original colors
    """
    width, height = image.size
    
    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = width / height
    if width > height:
        new_width = pixel_size
        new_height = int(pixel_size / aspect_ratio)
    else:
        new_height = pixel_size
        new_width = int(pixel_size * aspect_ratio)
    
    # Ensure minimum dimension is at least 1 pixel
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    # Reduce resolution and scale back up
    small_image = image.resize((new_width, new_height), Image.NEAREST)
    pixelated_image = small_image.resize((width, height), Image.NEAREST)
    
    return pixelated_image

def get_dominant_colors(image, num_colors):
    """
    Extract dominant colors by clustering and return median colors of each cluster.
    Uses MiniBatchKMeans which is more memory efficient.
    """
    # Convert image to numpy array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress KMeans warnings
        img_array = np.array(image.convert("RGB"))
        h, w, _ = img_array.shape
        
        # Reshape to 2D array of pixels (sample if too large)
        pixels = img_array.reshape((h * w, 3))
        
        # For large images, use a subset of pixels
        if len(pixels) > 10000:
            rng = np.random.default_rng()
            pixels = rng.choice(pixels, size=10000, replace=False)
        
        # Use MiniBatchKMeans which is more efficient
        kmeans = MiniBatchKMeans(n_clusters=num_colors, 
                                random_state=0,
                                batch_size=256,
                                compute_labels=True).fit(pixels)
        
        # Get cluster centers (already computed efficiently)
        cluster_colors = kmeans.cluster_centers_.astype(int)
        
        return [tuple(color) for color in cluster_colors]

def apply_median_palette(image, num_colors):

    """
    Quantize the image using clustered colors.
    More efficient implementation using direct cluster centers.
    """
    # Get the cluster colors
    palette_colors = get_dominant_colors(image, num_colors)
    
    # Create a palette image
    palette = Image.new("P", (1, 1))
    palette_data = [color for rgb in palette_colors for color in rgb]
    # Pad palette with black if needed
    palette_data += [0] * (256 * 3 - len(palette_data))
    palette.putpalette(palette_data)
    
    # Quantize the image using the custom palette
    quantized = image.convert("RGB").quantize(palette=palette, dither=Image.NONE)
    return quantized.convert("RGB")
