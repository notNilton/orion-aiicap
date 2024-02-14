from PIL import ImageTk
import tkinter as tk
import os

def load_images(*images):  # Aceita um número variável de imagens
    """
    Loads and displays multiple images in a Tkinter window, resized to a maximum of 600x600.
    """
    try:
        window = tk.Tk()
        window.title("Image Viewer")

        # Resize images if necessary
        max_size = (600, 600)

        # Para cada imagem, redimensiona e exibe
        for image in images:
            image.thumbnail(max_size)  # Redimensiona a imagem
            tk_image = ImageTk.PhotoImage(image)  # Converte para formato Tkinter

            label = tk.Label(window, image=tk_image)
            label.pack(side=tk.LEFT)
            label.image = tk_image  # Mantém uma referência para evitar garbage collection

        window.mainloop()
    except Exception as e:
        print(f"Error displaying images: {e}")      
          
def save_image(image, save_path, filename):
    """Saves a PIL Image object to the specified directory with a given filename."""
    try:
        os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist

        full_path = os.path.join(save_path, filename)  # Join path and filename

        image.save(full_path)
        print(f"Image saved successfully at {full_path}")
    except Exception as e:
        print(f"Error saving the image: {e}")

