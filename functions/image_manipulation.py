import numpy as np
from PIL import Image

def floyd_steinberg(image, levels=4):
    """
    Aplica Floyd-Steinberg em imagens RGB com paleta reduzida.
    """
    # Converte a imagem para um array NumPy
    pixels = np.array(image, dtype=float)
    width, height = image.size
    
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


def pixelate_image(image, scale_factor=0.1, colors=64):
    """
    Reduz a resolução da imagem, quantiza as cores e depois a expande, criando um efeito pixelizado.
    
    Args:
        image (PIL.Image): A imagem original.
        scale_factor (float): Fator de redução da resolução (0 < scale_factor < 1).
        colors (int): Número de cores para quantização (padrão: 16).
    
    Returns:
        PIL.Image: A imagem pixelizada com cores reduzidas.
    """
    # Calcula as novas dimensões
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Reduz a resolução da imagem
    small_image = image.resize((new_width, new_height), Image.NEAREST)
    
    # Quantiza as cores para o número especificado
    quantized_image = small_image.quantize(colors=colors)
    
    # Converte de volta para o modo RGB (necessário após quantização)
    quantized_image = quantized_image.convert("RGB")
    
    # Expande a imagem de volta para o tamanho original
    pixelated_image = quantized_image.resize((width, height), Image.NEAREST)
    
    return pixelated_image