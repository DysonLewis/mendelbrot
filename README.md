# Mandelbrot Set Generator

A high-performance, memory-efficient Mandelbrot set generator with multiple output formats, developed through iterative optimization for handling extremely high-resolution images.

## Files

- **`mendelbrot.py`** - Can generate much higher resolution images, at the cost of storage and time
- **`mandelbrot.cpp`** - C++ extension for fast Mandelbrot computation
- **`Makefile`** - Builds the C++ extension module

## Approximate Image File Sizes

### DeepZoom Pipeline (`mandelbrot.py`)

This was the only way I could get greater than around 5x resolution to work. Directly converting the FITS -> .png needed to load the entire file (even if it's processed in chunks)
This quickly eats up memory, also very difficult to open the .png file. It seemed to crash around 50% I tried to open it in an image viewing program.

**Default resolution for hw8_deepzoom.py is 4x** This should run and display the final image in ~5 minutes

**DeepZoom files are significantly larger than other formats** due to the pyramid structure generating tiles at multiple zoom levels.

| Multiplier | Resolution | Peak Storage* | Final Output |
|------------|------------|---------------|--------------|
| 1x | 7,680 × 10,240 | ~0.7 GB | ~0.6 GB |
| 10x | 76,800 × 102,400 | ~70 GB | ~58 GB |
| 20x | 153,600 × 204,800 | ~280 GB | ~233 GB |

*Peak storage includes temporary raw RGB file, intermediate TIFF, and final DeepZoom output during processing. After cleanup, only DeepZoom files remain.

**Storage breakdown per multiplier:**
- Raw RGB file: `(ny × nx × 3)` bytes (temporary, deleted after processing)
- TIFF file: ~30% of raw size (temporary, deleted after processing)  
- DeepZoom pyramid: ~25% of raw size (final output, permanent)

The max resolution I was able to test was 17x base, I would have gone higher but did not have the diskspace available (I need to clean my HST data lol), I think it took like 25 minutes to run on my machine.

## Requirements

```bash
# Python packages
pip install numpy astropy pyvips matplotlib

# System libraries (for pyvips)
# Ubuntu/Debian:
sudo apt-get install libvips-dev

# macOS:
brew install vips
```

## Usage

### Build the C++ Extension
```bash
make
```

### Generate FITS + PNG
```bash
python hw8_fits.py
```
Outputs:
- `output.fits` - Raw iteration count data
- `mandelbrot.png` - Colored visualization

### Generate Interactive DeepZoom Viewer
```bash
python hw8_deepzoom.py
```
Outputs:
- `mandelbrot_deepzoom_files/` - Tile pyramid directory
- `mandelbrot_deepzoom.dzi` - DeepZoom descriptor
- `mandelbrot_viewer.html` - Interactive web viewer
- Automatically opens in browser with local web server

## Key Optimizations

### Memory Management
- **Memory-mapped arrays**: Never load full images into RAM
- **Chunked processing**: Process data in small column/row strips
- **Streaming conversions**: Read from one file while writing to another
- **Immediate cleanup**: Delete intermediate data and call garbage collector

### Performance
- **C++ computation**: Core algorithm runs at compiled speeds
- **Multiprocessing**: Parallelizes work across all CPU cores
- **Batch processing**: Improves cache locality in tight loops
- **Queue management**: Limits queue size to prevent memory buildup

### Disk Usage
- **Direct pipelines**: Minimize intermediate files
- **Compressed formats**: Use deflate compression for TIFF, PNG tiles for DeepZoom
- **Cleanup**: Remove temporary files immediately after use

## Configuration

Edit parameters at the top of `hw8_deepzoom.py` or `hw8_fits.py`:

```python
max_iter = 300          # Iteration limit
xmin, xmax = -2.5, 1.   # X-axis domain
ymin, ymax = -1., 1.    # Y-axis domain
ny, nx = 7680, 10240    # Resolution (height, width)
ncol = 64               # Columns per chunk
```
At base resolution, max_iter = 100 is fine, above 10x consider bumping it to ~500.
For extreme resolutions (100+ megapixels), increase `ncol` to reduce queue overhead.
