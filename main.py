from PIL import Image, ImageTk
import tkinter as tk
from functions.image_manipulation import floyd_steinberg, pixelate_image
from functions.utils import save_image
import os

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        
        # Caminho da imagem original
        self.original_image_path = "./data/untreated/test-image.png"
        self.processed_image_path = "./data/treated/"
        
        # Carrega a imagem original
        try:
            self.input_image = Image.open(self.original_image_path)
            self.input_image_processed = self.input_image.convert("RGB")
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {self.original_image_path}")
            exit()
        
        # Exibe a imagem original na UI
        self.display_image(self.input_image_processed)
        
        # Cria os botões
        self.create_buttons()
    
    def display_image(self, image):
        """Exibe a imagem na interface gráfica."""
        # Redimensiona a imagem para caber na janela
        max_size = (600, 600)
        image.thumbnail(max_size)
        
        # Converte a imagem para o formato Tkinter
        self.tk_image = ImageTk.PhotoImage(image)
        
        # Exibe a imagem em um Label
        if hasattr(self, 'image_label'):
            self.image_label.config(image=self.tk_image)
        else:
            self.image_label = tk.Label(self.root, image=self.tk_image)
            self.image_label.pack()
        
        # Mantém uma referência para evitar garbage collection
        self.image_label.image = self.tk_image
    
    def create_buttons(self):
        """Cria os botões para aplicar os efeitos."""
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Botão para aplicar pixelização
        pixelate_button = tk.Button(
            button_frame, text="Aplicar Pixelização", command=self.apply_pixelation
        )
        pixelate_button.pack(side=tk.LEFT, padx=10)
        
        # Botão para aplicar dithering (Floyd-Steinberg)
        dither_button = tk.Button(
            button_frame, text="Aplicar Dithering", command=self.apply_dithering
        )
        dither_button.pack(side=tk.LEFT, padx=10)
    
    def apply_pixelation(self):
        """Aplica o efeito de pixelização e exibe a imagem."""
        # Sempre usa a imagem original como base
        pixelated_image = pixelate_image(self.input_image_processed, scale_factor=0.2)
        self.display_image(pixelated_image)
        
        # Salva a imagem pixelizada
        filename = os.path.basename(self.original_image_path)
        name, ext = os.path.splitext(filename)
        save_image(pixelated_image, self.processed_image_path, f"{name}_pixelated{ext}")
        print(f"Imagem pixelizada salva em {os.path.join(self.processed_image_path, f'{name}_pixelated{ext}')}")
    
    def apply_dithering(self):
        """Aplica o efeito de dithering (Floyd-Steinberg) e exibe a imagem."""
        # Sempre usa a imagem original como base
        dithered_image = floyd_steinberg(self.input_image_processed)
        self.display_image(dithered_image)
        
        # Salva a imagem com dithering
        filename = os.path.basename(self.original_image_path)
        name, ext = os.path.splitext(filename)
        save_image(dithered_image, self.processed_image_path, f"{name}_floyd{ext}")
        print(f"Imagem com dithering salva em {os.path.join(self.processed_image_path, f'{name}_floyd{ext}')}")


if __name__ == "__main__":
    # Cria a janela principal
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
