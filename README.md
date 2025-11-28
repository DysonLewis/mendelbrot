# Mandelbrot Set DeepZoom Generator

A high-performance, memory-efficient Mandelbrot set generator that creates interactive zoomable visualizations using the DeepZoom tile pyramid format. Optimized for extremely high-resolution rendering through parallel processing and streaming algorithms.

## Overview

This project originated as a homework assignment exploring parallel computing and image processing techniques. The original assignment focused on generating Mandelbrot set images in FITS and PNG formats. This standalone version extends that work with a complete DeepZoom implementation, enabling smooth interactive exploration of the fractal at arbitrary zoom levels.

**Original Assignment Features:**
- C++ extension for fast Mandelbrot computation
- Memory-mapped array processing for handling large images
- Multiprocessing with work queues
- FITS file output for scientific data
- Direct PNG generation

**What's New in This Version:**
- **Interactive DeepZoom viewer** with OpenSeadragon integration
- **Complete tile pyramid generation** with multiple zoom levels
- **Lanczos downsampling** for high-quality lower resolution tiles
- **Streaming tile processing** to minimize memory usage
- **Automated web server** for instant visualization
- **User-configurable resolution** via interactive prompts
- **Optimized C++ image processing** for vertical flips and tile splitting
- **Progress tracking** with tqdm progress bars

## Files

- **`mandelbrot.py`** - Main generator with DeepZoom pipeline
- **`mandelbrot.cpp`** - C++ extension for Mandelbrot computation
- **`image_processor.cpp`** - C++ extension for image transformations
- **`Makefile`** - Build script for C++ extensions

## Output Structure

The generator creates:
- `mandelbrot_deepzoom_files/` - Tile pyramid directory containing PNG tiles organized by zoom level
- `mandelbrot_deepzoom.dzi` - DeepZoom descriptor (XML metadata)
- `mandelbrot_viewer.html` - Interactive web viewer
- Automatic browser launch with local web server on port 8000

## Performance & Storage

### Approximate Storage Requirements

**Default resolution is 4x (30,720 × 40,960 pixels)**

DeepZoom creates a pyramid with tiles at multiple zoom levels, resulting in larger storage than single images but enabling smooth zooming.

| Scale | Resolution | Approximate Storage |
|-------|------------|-------------------|
| 1x | 7,680 × 10,240 | ~0.6 GB |
| 4x | 30,720 × 40,960 | ~10 GB |
| 8x | 61,440 × 81,920 | ~40 GB |
| 10x | 76,800 × 102,400 | ~58 GB |

Storage scales roughly with pixel count. The pyramid structure adds ~33% overhead compared to storing only the highest resolution level.

### Processing Time

On a typical multi-core system:
- 4x scale: ~5-10 minutes
- 10x scale: ~30-45 minutes
- 20x scale: ~2-3 hours

Time scales with pixel count and number of CPU cores available.

## Requirements

### Python Packages
```bash
pip install numpy matplotlib pillow tqdm
```

### C++ Compiler
- GCC or Clang with C++11 support
- Python development headers

On Ubuntu/Debian:
```bash
sudo apt-get install python3-dev build-essential
```

On macOS (with Xcode Command Line Tools):
```bash
xcode-select --install
```

## Installation & Usage

### 1. Build the C++ Extensions
```bash
make
```

### 2. Run the Generator
```bash
python mandelbrot.py
```

You'll be prompted to enter an image scale factor (press Enter for default 4x).

The generator will:
1. Compute the Mandelbrot set in parallel chunks
2. Assemble and save maximum resolution tiles
3. Build the pyramid of downsampled zoom levels
4. Generate the HTML viewer
5. Launch a local web server and open your browser

### 3. Explore the Visualization

Use your mouse to:
- **Drag** to pan around the set
- **Scroll** to zoom in/out
- Navigate smoothly across zoom levels

Press **Ctrl+C** in the terminal to stop the web server when done.

## Configuration

Key parameters in `mandelbrot.py`:

```python
max_iter = 300              # Iteration limit for escape test
color_reference = 100       # Color normalization reference
im_scale = 4                # Resolution multiplier (prompt at runtime)
xmin, xmax = -2.5, 1.       # Real axis bounds
ymin, ymax = -1., 1.        # Imaginary axis bounds
ny, nx = im_scale*7680, im_scale*10240  # Final resolution

TILE_SIZE = 256             # Tile dimensions (256 or 512)
ncol = 64                   # Number of computation chunks
```

### Tuning Tips

- **Higher resolution**: Increase `im_scale` when prompted (requires more time and storage)
- **Better detail at deep zoom**: Increase `max_iter` (300-500 for high resolutions)
- **Faster processing**: Set `TILE_SIZE = 512` (uses more memory but fewer tiles)
- **Lower memory usage**: Reduce `max_strip_workers` in the code (default: 8)

## Technical Details

### Key Optimizations

**Memory Management:**
- Chunks processed independently and saved to temporary files
- Strips assembled on-demand from memory-mapped chunk files
- Immediate cleanup of intermediate data
- Controlled worker pool sizes to limit memory footprint

**Performance:**
- C++ extensions for compute-intensive operations
- Multiprocessing across all available CPU cores
- Batch processing for cache efficiency
- Progress tracking without performance impact

**Image Quality:**
- Lanczos-3 resampling for downsampled pyramid levels
- Smooth coloring algorithm to eliminate banding
- Custom colormap with perceptually smooth gradients

### Color Palette

The default colormap transitions through:
Dark purple → Deep blue → Cyan → Green → Yellow → Orange → Red → Gray

Colors are normalized to a reference iteration count for consistency across different `max_iter` values.

## Troubleshooting

**"Module not found" error:**
Run `make` to compile the C++ extensions.

**Out of memory:**
- Reduce `im_scale` 
- Set `TILE_SIZE = 256` instead of 512
- Close other applications

**Port 8000 already in use:**
The generator will show an error but you can manually run:
```bash
python3 -m http.server 8000
```
Then open `http://localhost:8000/mandelbrot_viewer.html`

**Slow performance:**
- Check CPU usage during computation phase (should be near 100%)
- Ensure SSD is used for temporary files (not network drive)
- Consider reducing resolution for initial tests

## License

This project evolved from an academic assignment and is provided as-is for educational purposes.
