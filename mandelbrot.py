import numpy as np
import logging
import os
import sys
import multiprocessing as mp
import pyvips
from matplotlib.colors import LinearSegmentedColormap
import webbrowser
import http.server
import socketserver
import threading
import shutil
import gc
import math
from PIL import Image
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    import mandelbrot
    import image_processor
except ImportError:
    print("Error: mandelbrot or image_processor module not found")
    print("Please run 'python setup.py build_ext --inplace' to compile the C++ extensions")
    sys.exit(1)

_log = logging.getLogger('mandelbrot')

# Mandelbrot calculation parameters
max_iter = 300
r2_max = 1 << 16

# Reference iteration count for color normalization
# Colors will always match this scale regardless of max_iter
# This only changes the color, tbh I just liked how it looks at 100
color_reference = 100

# Get image scale factor from user
while True:
    try:
        im_scale_input = input("Enter image scale factor (default 4): ").strip()
        if im_scale_input == "":
            im_scale = 4
        else:
            im_scale = int(im_scale_input)
            if im_scale <= 0:
                print("Scale factor must be positive")
                continue
        break
    except ValueError:
        print("Please enter a valid integer")

# Define calculation domain and resolution
xmin, xmax = -2.5, 1.
ymin, ymax = -1., 1.
ny, nx = im_scale*7680, im_scale*10240
x = np.linspace(xmin, xmax, nx, endpoint=True)
y = np.linspace(ymin, ymax, ny, endpoint=True)

# DeepZoom tile size
TILE_SIZE = 512
TILE_OVERLAP = 0

# Setup for chunking the x-axis into columns
ncol = 64
fx = nx//ncol
nc = fx + (fx*ncol < nx)
bx = np.arange(nc, dtype=int)*ncol
ex = np.clip((np.arange(nc, dtype=int)+1)*ncol, 0, nx)


def worker(input, output, lut_shared, color_max_shared):
    '''Worker process that computes Mandelbrot values and converts to RGB'''
    lut = np.frombuffer(lut_shared, dtype=np.uint8).reshape(256, 3)
    color_max = color_max_shared.value
    
    for i, args in iter(input.get, 'STOP'):
        coldata = mandelbrot.calc_val(*args)
        
        # Convert to RGB in worker to reduce data transfer
        chunk_normalized = np.clip((coldata / color_max) * 255, 0, 255).astype(np.uint8)
        rgb_chunk = lut[chunk_normalized]
        
        output.put((i, rgb_chunk))
        del coldata, chunk_normalized


def feeder(input):
    '''Feeder process that creates work chunks and queues them for workers'''
    for i in range(nc):
        xx, yy = np.meshgrid(x[bx[i]:ex[i]], y)
        args = (xx, yy)
        input.put((i, args), True)
    _log.debug('feeder finished')


def create_dzi_file(width, height, tile_size, tile_overlap, dzi_path):
    '''Create the .dzi metadata file for DeepZoom'''
    dzi_content = f'''<?xml version="1.0" encoding="utf-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="png"
       Overlap="{tile_overlap}"
       TileSize="{tile_size}">
    <Size Height="{height}" Width="{width}"/>
</Image>'''
    
    with open(dzi_path, 'w') as f:
        f.write(dzi_content)


def save_tile(tile_data, level, col, row, tiles_dir):
    '''Save a single tile as PNG'''
    level_dir = os.path.join(tiles_dir, str(level))
    os.makedirs(level_dir, exist_ok=True)
    
    tile_path = os.path.join(level_dir, f'{col}_{row}.png')
    
    img = Image.fromarray(tile_data, mode='RGB')
    img.save(tile_path, 'PNG', optimize=False)


def get_level_dimensions(width, height, level, max_level):
    '''Calculate dimensions at a given pyramid level'''
    scale = 2 ** (max_level - level)
    level_width = max(1, int(math.ceil(width / scale)))
    level_height = max(1, int(math.ceil(height / scale)))
    return level_width, level_height


def process_strip_worker(args):
    '''Worker function to process one strip and save tiles'''
    strip_idx, strip_y_start, strip_y_end, strip_height, nx, bx, ex, nc, tiles_dir, max_level, strips_per_height = args
    
    # Build strip by loading only needed portions from chunks
    strip_buffer = np.zeros((strip_height, nx, 3), dtype=np.uint8)
    
    for i in range(nc):
        chunk_file = os.path.join(tiles_dir, f'temp_chunk_{i:04d}.npy')
        # Memory-map the chunk file to avoid loading entire chunk
        chunk_mmap = np.load(chunk_file, mmap_mode='r')
        # Copy only the needed strip portion
        strip_buffer[:, bx[i]:ex[i], :] = chunk_mmap[strip_y_start:strip_y_end, :, :]
        del chunk_mmap
    
    # Flip vertically using C++ extension
    strip_buffer = image_processor.flip_vertical(strip_buffer)
    
    # Split into tiles using C++ extension
    tiles = image_processor.split_into_tiles(strip_buffer, TILE_SIZE)
    del strip_buffer
    
    # Save tiles
    tiles_per_width = len(tiles)
    tile_row = strips_per_height - 1 - strip_idx
    
    for tile_x_idx, tile_data in enumerate(tiles):
        save_tile(tile_data, max_level, tile_x_idx, tile_row, tiles_dir)
    
    return strip_idx


def downsample_tile_worker(args):
    '''Worker function to create one downsampled tile from 2x2 source tiles'''
    level, tile_col, tile_row, tiles_dir, source_level = args
    
    source_tile_col = tile_col * 2
    source_tile_row = tile_row * 2
    
    # Load up to 4 source tiles and combine into 512x512 image
    combined = np.zeros((TILE_SIZE * 2, TILE_SIZE * 2, 3), dtype=np.uint8)
    
    for dy in range(2):
        for dx in range(2):
            src_col = source_tile_col + dx
            src_row = source_tile_row + dy
            
            source_tile_path = os.path.join(tiles_dir, str(source_level), f'{src_col}_{src_row}.png')
            
            if os.path.exists(source_tile_path):
                img = Image.open(source_tile_path)
                tile_array = np.array(img)
                
                y_start = dy * TILE_SIZE
                x_start = dx * TILE_SIZE
                combined[y_start:y_start + tile_array.shape[0], 
                        x_start:x_start + tile_array.shape[1], :] = tile_array
    
    # Downsample using C++ Lanczos implementation
    downsampled = image_processor.downsample_tile(combined)
    del combined
    
    save_tile(downsampled, level, tile_col, tile_row, tiles_dir)
    
    return (level, tile_col, tile_row)


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        )
    
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    _log.info(f'Generating Mandelbrot set at {ny} x {nx} resolution')
    
    # Setup colormap early so workers can use it
    # VScode is great I could just select a color on the graph thingy
    colors = ["#10001F", "#1A0E36", "#001E71", "#007D7D", "#006C7F", 
              "#00B129", "#F2FF00", "#FF6600", "#D60000", "#757575FF"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('mandelbrot', colors, N=n_bins)
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    color_max = float(color_reference)
    
    # Share colormap with workers via shared memory
    lut_shared = mp.Array('B', lut.flatten(), lock=False)
    color_max_shared = mp.Value('d', color_max, lock=False)
    
    # Setup DeepZoom directory structure
    dz_dir = os.path.join(script_dir, 'mandelbrot_deepzoom')
    tiles_dir = dz_dir + '_files'
    
    if os.path.exists(tiles_dir):
        _log.info(f'Removing existing DeepZoom directory: {tiles_dir}')
        shutil.rmtree(tiles_dir)
    
    os.makedirs(tiles_dir, exist_ok=True)
    
    # Calculate max pyramid level
    max_level = int(math.ceil(math.log(max(nx, ny), 2)))
    _log.info(f'DeepZoom pyramid will have {max_level + 1} levels')
    
    # Setup multiprocessing queues and processes
    n_process = mp.cpu_count()
    n_max = n_process*2
    inqueue = mp.Queue(n_max)
    outqueue = mp.Queue(n_max)

    # Start worker processes to compute Mandelbrot chunks
    for i in range(n_process):
        mp.Process(target=worker, args=(inqueue, outqueue, lut_shared, color_max_shared)).start()

    # Start feeder process to generate work
    feedp = mp.Process(target=feeder, args=(inqueue,))
    feedp.start()
    
    _log.info('Phase 1: Computing chunks and writing max-resolution tiles')
    
    # We'll build the full-resolution image in horizontal strips to save tiles
    # Allocate buffer for one strip of tiles (height = TILE_SIZE)
    strips_per_height = int(math.ceil(ny / TILE_SIZE))
    current_strip = np.zeros((TILE_SIZE, nx, 3), dtype=np.uint8)
    strip_fill_height = 0
    current_strip_idx = 0
    
    # Process chunks and accumulate into strips
    chunk_files = {}
    with tqdm(total=nc, desc="Computing chunks", unit="chunk") as pbar:
        for j in range(nc):
            i, rgb_chunk = outqueue.get()
            
            # Store chunk in the appropriate position
            # For now, save chunks to temp files since they come out of order
            chunk_file = os.path.join(tiles_dir, f'temp_chunk_{i:04d}.npy')
            np.save(chunk_file, rgb_chunk)
            chunk_files[i] = chunk_file
            
            del rgb_chunk
            pbar.update(1)
            
            if (j + 1) % 10 == 0:
                gc.collect()
    
    # Clean up worker processes
    _log.debug('received all chunks; killing workers')
    for i in range(n_process):
        inqueue.put('STOP')
    
    _log.debug('waiting for feeder to finish')
    feedp.join(1.)
    
    _log.info('All chunks computed, now assembling and saving tiles')
    
    level_dir = os.path.join(tiles_dir, str(max_level))
    os.makedirs(level_dir, exist_ok=True)
    
    # Build full image strip by strip and save tiles in parallel
    strips_to_process = []
    for strip_idx in range(strips_per_height):
        strip_y_start = strip_idx * TILE_SIZE
        strip_y_end = min((strip_idx + 1) * TILE_SIZE, ny)
        strip_height = strip_y_end - strip_y_start
        
        strips_to_process.append((
            strip_idx, strip_y_start, strip_y_end, strip_height, 
            nx, bx, ex, nc, tiles_dir, max_level, strips_per_height
        ))
    
    # Process strips in parallel
    max_strip_workers = mp.cpu_count()
    _log.info(f'Processing {strips_per_height} strips with {max_strip_workers} workers')
    
    with mp.Pool(max_strip_workers) as pool:
        with tqdm(total=strips_per_height, desc="Processing strips", unit="strip") as pbar:
            for result in pool.imap(process_strip_worker, strips_to_process):
                pbar.update(1)
        pool.close()
        pool.join()
    
    # Clean up temporary chunk files
    _log.info('Cleaning up temporary chunk files')
    for i in range(nc):
        chunk_file = os.path.join(tiles_dir, f'temp_chunk_{i:04d}.npy')
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
    
    _log.info('Phase 2: Building pyramid levels')
    
    # Build pyramid from max_level down to 0
    for level in range(max_level - 1, -1, -1):
        source_level = level + 1
        
        level_width, level_height = get_level_dimensions(nx, ny, level, max_level)
        source_width, source_height = get_level_dimensions(nx, ny, source_level, max_level)
        
        tiles_wide = int(math.ceil(level_width / TILE_SIZE))
        tiles_high = int(math.ceil(level_height / TILE_SIZE))
        
        _log.debug(f"Building level {level}: {level_width}x{level_height} ({tiles_wide}x{tiles_high} tiles)")
        
        level_dir = os.path.join(tiles_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)
        
        # Create list of all tiles to process for this level
        tiles_to_process = []
        for tile_row in range(tiles_high):
            for tile_col in range(tiles_wide):
                tiles_to_process.append((level, tile_col, tile_row, tiles_dir, source_level))
        
        # Process tiles in parallel
        max_tile_workers = mp.cpu_count() * 2
        
        with mp.Pool(max_tile_workers) as pool:
            with tqdm(total=len(tiles_to_process), desc=f"Level {level}", unit="tile") as pbar:
                for result in pool.imap_unordered(downsample_tile_worker, tiles_to_process):
                    pbar.update(1)
            pool.close()
            pool.join()
    
    # Create .dzi file
    dzi_path = dz_dir + '.dzi'
    create_dzi_file(nx, ny, TILE_SIZE, TILE_OVERLAP, dzi_path)
    _log.info(f'Created .dzi file: {dzi_path}')
    
    # Generate HTML viewer for DeepZoom
    # Lowk had to look this up, idk if it's any good
    _log.info('Generating HTML viewer')
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mandelbrot Set Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #000;
        }}
        #viewer {{
            width: 100vw;
            height: 100vh;
            background: #000;
        }}
        .info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: #fff;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="info">
        Mandelbrot Set ({nx} × {ny} pixels, {max_iter} iterations)<br>
        Use mouse wheel to zoom, drag to pan
    </div>
    <div id="viewer"></div>
    <script>
        OpenSeadragon({{
            id: "viewer",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: "mandelbrot_deepzoom.dzi",
            showNavigationControl: true,
            navigationControlAnchor: OpenSeadragon.ControlAnchor.TOP_RIGHT,
            animationTime: 0.5,
            blendTime: 0.1,
            constrainDuringPan: false,
            maxZoomPixelRatio: 1000,
            minZoomLevel: 0.8,
            visibilityRatio: 1,
            zoomPerScroll: 1.2,
            timeout: 120000
        }});
    </script>
</body>
</html>"""

    html_fn = os.path.join(script_dir, 'mandelbrot_viewer.html')
    with open(html_fn, 'w') as f:
        f.write(html_content)

    _log.info('DeepZoom pyramid saved successfully')
    _log.info(f'HTML viewer created: {html_fn}')
    _log.info(f'Open {html_fn} in your browser to view the Mandelbrot set')

    # Start local web server and open browser
    _log.info('Starting local web server')
    PORT = 8000
    
    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
    
    os.chdir(script_dir)
    Handler = QuietHTTPRequestHandler
    
    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
        _log.info(f'Web server running at http://localhost:{PORT}')
        
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        
        url = f'http://localhost:{PORT}/mandelbrot_viewer.html'
        _log.info(f'Opening browser to {url}')
        webbrowser.open(url)
        
        _log.info('Press Ctrl+C to stop the server and exit')
        
        try:
            server_thread.join()
        except KeyboardInterrupt:
            _log.info('Shutting down server')
            httpd.shutdown()
            
    except OSError as e:
        _log.warning(f'Could not start server on port {PORT}: {e}')
        _log.info(f'You can manually run: python3 -m http.server {PORT}')
        _log.info(f'Then open: http://localhost:{PORT}/mandelbrot_viewer.html')