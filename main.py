from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from functions.image_processor import (
    floyd_steinberg, 
    pixelate_image, 
    apply_median_palette,
    fix_image
)
from functions.utils import save_image
from functions.strategies import Strategies
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
        
        # Create image display frame
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()
        
        # Display the original image
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        self.display_image(self.current_image)
        
        # Create buttons
        self.create_buttons()
    
    def display_image(self, image):
        """Display the image in the GUI without modifying the original."""
        # Create a copy for display
        display_image = image.copy()
        
        # Calculate maximum display size while maintaining aspect ratio
        max_width, max_height = 800, 600
        width, height = display_image.size
        ratio = min(max_width/width, max_height/height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # Resize for display
        display_image = display_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to Tkinter format
        self.tk_image = ImageTk.PhotoImage(display_image)
        
        # Update the image label
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image  # Keep a reference
    
    def create_buttons(self):
        """Create filter application buttons."""
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Pixelate button
        tk.Button(
            button_frame, text="Apply Pixelation", 
            command=self.apply_pixelation
        ).pack(side=tk.LEFT, padx=5)
        
        # Dithering button
        tk.Button(
            button_frame, text="Apply Dithering", 
            command=self.apply_dithering
        ).pack(side=tk.LEFT, padx=5)
        
        # Median Palette button
        tk.Button(
            button_frame, text="Median Palette", 
            command=self.apply_median_palette
        ).pack(side=tk.LEFT, padx=5)
        
        # Color Fix button
        tk.Button(
            button_frame, text="Color Fix", 
            command=self.apply_color_fix
        ).pack(side=tk.LEFT, padx=5)
        
        # Strategy selection
        self.strategy_var = tk.StringVar(value=Strategies.AVERAGE.name)
        tk.OptionMenu(
            button_frame, 
            self.strategy_var, 
            *[s.name for s in Strategies]
        ).pack(side=tk.LEFT, padx=5)
        
        # Reset button
        tk.Button(
            button_frame, text="Reset", 
            command=self.reset_to_original
        ).pack(side=tk.LEFT, padx=5)
    
    def apply_pixelation(self):
        """Apply pixelation to the current image."""
        try:
            pixelated_image = pixelate_image(self.current_image)
            self.current_image = pixelated_image
            self.display_image(self.current_image)
            self.save_image("pixelated")
        except Exception as e:
            print(f"Error applying pixelation: {e}")
    
    def apply_dithering(self):
        """Apply dithering to the current image."""
        try:
            dithered_image = floyd_steinberg(self.current_image)
            self.current_image = dithered_image
            self.display_image(self.current_image)
            self.save_image("floyd")
        except Exception as e:
            print(f"Error applying dithering: {e}")
    
    def apply_median_palette(self):
        """Apply median color palette to the current image."""
        try:
            median_image = apply_median_palette(self.current_image, num_colors=32)
            self.current_image = median_image
            self.display_image(self.current_image)
            self.save_image("median")
        except Exception as e:
            print(f"Error applying median palette: {e}")
    
    def apply_color_fix(self):
        """Apply color fix with selected strategy."""
        try:
            # Convert to RGBA if not already
            if self.current_image.mode != 'RGBA':
                img_data = np.array(self.current_image.convert("RGBA"))
            else:
                img_data = np.array(self.current_image)
                
            strategy = Strategies[self.strategy_var.get()]
            
            processed, h, w = fix_image(
                img_data,
                self.current_image.height,
                self.current_image.width,
                out_pix_width=4,
                out_pix_height=4,
                strategy=strategy
            )
            
            # Convert to proper image format
            if processed.dtype != np.uint8:
                processed = processed.astype(np.uint8)
            
            # Convert back to PIL Image
            if processed.ndim == 3 and processed.shape[2] == 4:
                self.current_image = Image.fromarray(processed, 'RGBA')
            elif processed.ndim == 3 and processed.shape[2] == 3:
                self.current_image = Image.fromarray(processed, 'RGB')
            else:
                self.current_image = Image.fromarray(processed)
            
            self.display_image(self.current_image)
            self.save_image(f"colorfix_{strategy.name.lower()}")
        except Exception as e:
            print(f"Error applying color fix: {e}")
    
    def reset_to_original(self):
        """Reset the current image to the original state."""
        self.current_image = self.original_image.copy()
        self.display_image(self.current_image)
    
    def save_image(self, suffix):
        """Helper method to save processed images."""
        try:
            filename = os.path.basename(self.original_image_path)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(self.processed_image_path, f"{name}_{suffix}{ext}")
            # Make sure the directory exists
            os.makedirs(self.processed_image_path, exist_ok=True)
            # Call save_image with correct arguments
            save_image(self.current_image, self.processed_image_path, f"{name}_{suffix}{ext}")
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()