# main.py

from PIL import Image, ImageTk
import tkinter as tk
from functions.image_manipulation import (
    floyd_steinberg, 
    pixelate_image, 
    apply_median_palette,
)
from functions.utils import save_image
import os

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        
        # Image paths
        self.original_image_path = "./data/untreated/medieval-landscape.png"
        self.processed_image_path = "./data/treated/"
        
        # Load and store the original image in RGB mode
        try:
            self.input_image = Image.open(self.original_image_path)
            self.original_image = self.input_image.convert("RGB")  # Permanent reference to the original
            self.current_image = self.original_image.copy()        # Track current image state
        except FileNotFoundError:
            print(f"Error: File not found at {self.original_image_path}")
            exit()
        
        # Display the original image
        self.display_image(self.current_image)
        
        # Create buttons
        self.create_buttons()
    
    def display_image(self, image):
        """Display the image in the GUI without modifying the original."""
        display_image = image.copy()
        max_size = (600, 600)
        display_image.thumbnail(max_size)
        
        # Convert to Tkinter format
        self.tk_image = ImageTk.PhotoImage(display_image)
        
        # Update or create the image label
        if hasattr(self, 'image_label'):
            self.image_label.config(image=self.tk_image)
        else:
            self.image_label = tk.Label(self.root, image=self.tk_image)
            self.image_label.pack()
        
        self.image_label.image = self.tk_image  # Keep a reference
    
    def create_buttons(self):
        """Create filter application buttons."""
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Pixelate button
        pixelate_button = tk.Button(
            button_frame, text="Apply Pixelation", command=self.apply_pixelation
        )
        pixelate_button.pack(side=tk.LEFT, padx=10)
        
        # Dithering button
        dither_button = tk.Button(
            button_frame, text="Apply Dithering", command=self.apply_dithering
        )
        dither_button.pack(side=tk.LEFT, padx=10)
        
        # Median Palette button (changed from Vibrant)
        median_button = tk.Button(
            button_frame, text="Apply Median Palette", command=self.apply_median_palette
        )
        median_button.pack(side=tk.LEFT, padx=10)

        # Reset to Original button
        reset_button = tk.Button(
            button_frame, text="Reset to Original", command=self.reset_to_original
        )
        reset_button.pack(side=tk.LEFT, padx=10)
    
    def apply_pixelation(self):
        """Apply pixelation to the current image."""
        pixelated_image = pixelate_image(self.current_image)
        self.current_image = pixelated_image  # Update current image
        self.display_image(self.current_image)
        
        # Save the processed image
        filename = os.path.basename(self.original_image_path)
        name, ext = os.path.splitext(filename)
        save_image(self.current_image, self.processed_image_path, f"{name}_pixelated{ext}")
        print(f"Pixelated image saved to {os.path.join(self.processed_image_path, f'{name}_pixelated{ext}')}")
    
    def apply_dithering(self):
        """Apply dithering to the current image."""
        dithered_image = floyd_steinberg(self.current_image)
        self.current_image = dithered_image  # Update current image
        self.display_image(self.current_image)
        
        # Save the processed image
        filename = os.path.basename(self.original_image_path)
        name, ext = os.path.splitext(filename)
        save_image(self.current_image, self.processed_image_path, f"{name}_floyd{ext}")
        print(f"Dithered image saved to {os.path.join(self.processed_image_path, f'{name}_floyd{ext}')}")
    
    def reset_to_original(self):
        """Reset the current image to the original state."""
        self.current_image = self.original_image.copy()
        self.display_image(self.current_image)
        print("Image reset to original state.")


    def apply_median_palette(self):
        """Apply median color palette to the current image."""
        median_image = apply_median_palette(self.current_image, num_colors=32)
        self.current_image = median_image
        self.display_image(self.current_image)
        
        # Save the processed image
        filename = os.path.basename(self.original_image_path)
        name, ext = os.path.splitext(filename)
        save_image(self.current_image, self.processed_image_path, f"{name}_median{ext}")
        print(f"Median palette image saved to {os.path.join(self.processed_image_path, f'{name}_median{ext}')}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()



