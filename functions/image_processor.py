import numpy as np
from typing import List, Tuple
from collections import defaultdict
from functions.strategies import Strategies

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