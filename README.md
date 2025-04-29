
# Artificial 

A Python-based GUI application for applying various image processing techniques including pixelation, dithering (Floyd-Steinberg), and median palette reduction.

## Features

- **Pixelation**: Reduce image resolution and scale back up for a blocky pixel effect
- **Floyd-Steinberg Dithering**: Apply error-diffusion dithering to reduce color banding
- **Median Palette Reduction**: Quantize image to a reduced color palette using k-means clustering
- **Interactive GUI**: Built with Tkinter for easy operation
- **Image Comparison**: View original and processed images side-by-side
- **Auto-saving**: Processed images are automatically saved with descriptive filenames

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/notNilton/orion-aiicap.git
   cd image-processor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install them manually:
   ```bash
   pip install numpy opencv-python matplotlib numba scikit-learn Pillow
   ```

## Usage

1. Place your input images in the `./data/untreated/` directory
2. Run the application:
   ```bash
   python main.py
   ```
3. Use the buttons in the GUI to apply different effects:
   - **Apply Pixelation**: Creates a blocky pixel art effect
   - **Apply Dithering**: Applies Floyd-Steinberg dithering
   - **Apply Median Palette**: Reduces colors using k-means clustering
   - **Reset to Original**: Reverts to the original image

Processed images are automatically saved to `./data/treated/` with appropriate suffixes.

## File Structure

```
image-processor/
├── data/
│   ├── untreated/          # Input images go here
│   └── treated/            # Processed images are saved here
├── functions/
│   ├── image_manipulation.py  # Core image processing functions
│   └── utils.py               # Utility functions
├── main.py                 # Main application GUI
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Technical Details

### Implemented Algorithms

1. **Floyd-Steinberg Dithering**:
   - Error-diffusion dithering algorithm
   - Reduces color banding while maintaining perceived color depth
   - Implemented with NumPy for efficient pixel operations

2. **Pixelation**:
   - Resizes image to lower resolution and scales back up
   - Uses nearest-neighbor interpolation to maintain blocky appearance

3. **Median Palette Reduction**:
   - Uses MiniBatchKMeans clustering to identify dominant colors
   - Quantizes image to the clustered color palette
   - More efficient than standard k-means for large images

### Dependencies

- Python 3.7+
- Pillow (PIL Fork) - Image processing
- NumPy - Numerical operations
- scikit-learn - K-means clustering
- Tkinter - GUI (usually included with Python)
