import numpy as np
from typing import List, Tuple
from collections import defaultdict
from functions.strategies import Strategies
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import warnings

Pixel = List[int]

def fix_image(image_data: np.ndarray, 
             height: int, 
             width: int, 
             out_pix_width: int, 
             out_pix_height: int, 
             strategy: Strategies, 
             tolerance: int = 1, 
             shrink_output: bool = False) -> Tuple[np.ndarray, int, int]:   
    """
    Process an image using block-based color strategies
    
    Args:
        image_data: Input image as numpy array (height, width, channels)
        height: Original image height
        width: Original image width
        out_pix_width: Output pixel block width
        out_pix_height: Output pixel block height
        strategy: Color processing strategy
        tolerance: Color matching tolerance
        shrink_output: Whether to shrink output to block size
    
    Returns:
        Tuple of (processed_image, new_height, new_width)
    """
    # Adjust dimensions to be divisible by block size
    adjusted_width = (width // out_pix_width) * out_pix_width
    adjusted_height = (height // out_pix_height) * out_pix_height
    
    # Split into blocks
    blocks = []
    for y in range(0, adjusted_height, out_pix_height):
        for x in range(0, adjusted_width, out_pix_width):
            block = image_data[y:y+out_pix_height, x:x+out_pix_width]
            blocks.append(block.reshape(-1, block.shape[-1]))
    
    # Process each block
    processed_blocks = []
    for block in blocks:
        if strategy == Strategies.MAJORITY:
            color, _ = get_majority_color(block, tolerance)
            processed_block = np.tile(color, (len(block), 1))
        elif strategy == Strategies.AVERAGE:
            color = get_average_of_colors(block)
            processed_block = np.tile(color, (len(block), 1))
        elif strategy == Strategies.HARMONIC:
            color = get_harmonic_mean_of_colors(block)
            processed_block = np.tile(color, (len(block), 1))
        elif strategy == Strategies.GEOMETRIC:
            color = get_geometric_mean_of_colors(block)
            processed_block = np.tile(color, (len(block), 1))
        elif strategy == Strategies.MIDRANGE:
            color = get_midrange_of_colors(block)
            processed_block = np.tile(color, (len(block), 1))
        elif strategy == Strategies.QUADRATIC:
            color = get_quadratic_mean_of_colors(block)
            processed_block = np.tile(color, (len(block), 1))
        elif strategy == Strategies.CUBIC:
            color = get_cubic_mean_of_colors(block)
            processed_block = np.tile(color, (len(block), 1))
        else:  # Algorithm strategies
            color, occurrences = get_majority_color(block, tolerance)
            coverage = occurrences / len(block)
            if coverage >= strategy.value:
                processed_block = np.tile(color, (len(block), 1))
            else:
                avg_color = get_average_of_colors(block)
                processed_block = np.tile(avg_color, (len(block), 1))
        
        processed_blocks.append(processed_block)
    
    if shrink_output:
        # Create reduced size image (one pixel per block)
        out_height = adjusted_height // out_pix_height
        out_width = adjusted_width // out_pix_width
        out_data = np.array([block[0] for block in processed_blocks])
        out_data = out_data.reshape(out_height, out_width, -1)
    else:
        # Reconstruct full size image
        out_height = adjusted_height
        out_width = adjusted_width
        out_data = np.zeros((adjusted_height * adjusted_width, image_data.shape[2]))
        
        block_idx = 0
        for y in range(0, adjusted_height, out_pix_height):
            for x in range(0, adjusted_width, out_pix_width):
                block = processed_blocks[block_idx]
                idx = 0
                for by in range(out_pix_height):
                    for bx in range(out_pix_width):
                        pos = (y + by) * adjusted_width + (x + bx)
                        out_data[pos] = block[idx]
                        idx += 1
                block_idx += 1
        
        out_data = out_data.reshape(adjusted_height, adjusted_width, -1)
    
    out_data = np.clip(out_data, 0, 255).astype(np.uint8)
    return out_data, out_height, out_width

# Color calculation functions
def get_average_of_colors(pixels: np.ndarray) -> Pixel:
    return np.round(np.mean(pixels, axis=0)).astype(int).tolist()

def get_harmonic_mean_of_colors(pixels: np.ndarray) -> Pixel:
    harmonic = len(pixels) / np.sum(1.0 / np.maximum(pixels, 1e-6), axis=0)
    return np.round(harmonic).astype(int).tolist()

def get_geometric_mean_of_colors(pixels: np.ndarray) -> Pixel:
    geometric = np.prod(np.maximum(pixels, 1), axis=0) ** (1.0 / len(pixels))
    return np.round(geometric).astype(int).tolist()

def get_midrange_of_colors(pixels: np.ndarray) -> Pixel:
    min_vals = np.min(pixels, axis=0)
    max_vals = np.max(pixels, axis=0)
    return np.round((min_vals + max_vals) / 2).astype(int).tolist()

def get_quadratic_mean_of_colors(pixels: np.ndarray) -> Pixel:
    quadratic = np.sqrt(np.mean(np.square(pixels), axis=0))
    return np.round(quadratic).astype(int).tolist()

def get_cubic_mean_of_colors(pixels: np.ndarray) -> Pixel:
    cubic = np.cbrt(np.mean(np.power(pixels, 3), axis=0))
    return np.round(cubic).astype(int).tolist()

def get_majority_color(pixels: np.ndarray, tolerance: int = 1) -> Tuple[Pixel, int]:
    color_counts = defaultdict(int)
    
    for pixel in pixels:
        found = False
        for color in color_counts:
            if all(abs(pixel[i] - color[i]) <= tolerance for i in range(len(pixel))):
                color_counts[color] += 1
                found = True
                break
        if not found:
            color_counts[tuple(pixel)] = 1
    
    majority_color = max(color_counts.items(), key=lambda x: x[1])
    return list(majority_color[0]), majority_color[1]

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
