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
    print("Please run 'make' to compile the C++ extensions")
    sys.exit(1)

_log = logging.getLogger('mandelbrot')

# Mandelbrot calculation parameters
max_iter = 750
r2_max = 1 << 18

# Reference iteration count for color normalization
# Colors will always match this scale regardless of max_iter
# This only changes the color, tbh I just liked how it looks at 100
color_reference = 100

# Get image scale factor from user with time estimate
def estimate_time(scale):
    '''
    Estimate processing time based on empirical formula: t = 12.977 × x^1.804
    Function fit based on my machine timing running at 750 iterations 
    '''
    time_seconds = 12.977 * (scale ** 1.804)
    time_minutes = time_seconds / 60
    
    if time_minutes < 1:
        return f"{time_seconds:.0f} seconds"
    elif time_minutes < 60:
        return f"{time_minutes:.1f} minutes"
    else:
        hours = time_minutes / 60
        return f"{hours:.1f} hours"

def estimate_storage(scale):
    '''
    Estimate storage requirements based on scale
    Peak storage: one strip of chunks (256 pixels high × width × 64 chunks × 3 bytes)
    Empirical formula from testing: peak ≈ 0.075 × scale GB
    '''
    peak_gb = 0.075 * scale
    
    # Final DeepZoom storage (compressed PNGs)
    # Power law formula fitted from empirical data
    final_mb = 9.58017 * (scale ** 1.69713)
    
    if peak_gb < 1:
        peak_str = f"{peak_gb * 1024:.0f} MB"
    else:
        peak_str = f"{peak_gb:.1f} GB"
    
    if final_mb < 1024:
        final_str = f"{final_mb:.0f} MB"
    else:
        final_str = f"{final_mb / 1024:.1f} GB"
    
    return peak_str, final_str

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
        
        # Calculate and display estimates
        ny_calc = im_scale * 7680
        nx_calc = im_scale * 10240
        estimated_time = estimate_time(im_scale)
        peak_storage, final_storage = estimate_storage(im_scale)
        
        print(f"\nScale {im_scale}x:")
        print(f"  Resolution: {ny_calc:,} × {nx_calc:,} pixels")
        print(f"  Estimated time: {estimated_time}")
        print(f"  Peak temp storage: {peak_storage}")
        print(f"  Final DeepZoom size: {final_storage}")
        
        # Confirm with user
        confirm = input("Continue with this scale? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            break
        else:
            print("Let's try a different scale.\n")
            continue
            
    except ValueError:
        print("Please enter a valid integer")

print(f"\nStarting generation at {im_scale}x scale...")

# Define calculation domain and resolution
xmin, xmax = -2.5, 1.
ymin, ymax = -1., 1.
ny, nx = im_scale*7680, im_scale*10240
x = np.linspace(xmin, xmax, nx, endpoint=True)
y = np.linspace(ymin, ymax, ny, endpoint=True)

# DeepZoom tile size
TILE_SIZE = 256
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


def feeder(input, strip_y_start, strip_y_end):
    '''Feeder process that creates work chunks for a specific strip'''
    for i in range(nc):
        xx, yy = np.meshgrid(x[bx[i]:ex[i]], y[strip_y_start:strip_y_end])
        args = (xx, yy)
        input.put((i, args), True)


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
    
    # Calculate strip parameters
    strips_per_height = int(math.ceil(ny / TILE_SIZE))
    
    _log.info(f'Processing image in {strips_per_height} strips')
    
    # Setup multiprocessing queues and processes
    n_process = mp.cpu_count()
    n_max = n_process*2
    
    level_dir = os.path.join(tiles_dir, str(max_level))
    os.makedirs(level_dir, exist_ok=True)
    
    # Process one strip at a time
    with tqdm(total=strips_per_height, desc="Processing strips", unit="strip", position=0) as strip_pbar:
        for strip_idx in range(strips_per_height):
            strip_y_start = strip_idx * TILE_SIZE
            strip_y_end = min((strip_idx + 1) * TILE_SIZE, ny)
            strip_height = strip_y_end - strip_y_start
            
            # Create fresh queues for this strip
            inqueue = mp.Queue(n_max)
            outqueue = mp.Queue(n_max)
            
            # Start worker processes for this strip
            workers = []
            for i in range(n_process):
                p = mp.Process(target=worker, args=(inqueue, outqueue, lut_shared, color_max_shared))
                p.start()
                workers.append(p)
            
            # Start feeder process for this strip
            feedp = mp.Process(target=feeder, args=(inqueue, strip_y_start, strip_y_end))
            feedp.start()
            
            # Collect chunks and save as temporary .npy files with nested progress bar
            chunk_files = []
            with tqdm(total=nc, desc=f"  Strip {strip_idx+1}/{strips_per_height} chunks", 
                     unit="chunk", position=1, leave=False) as chunk_pbar:
                for j in range(nc):
                    i, rgb_chunk = outqueue.get()
                    
                    # Save chunk temporarily
                    chunk_file = os.path.join(tiles_dir, f'temp_chunk_{i:04d}.npy')
                    np.save(chunk_file, rgb_chunk)
                    chunk_files.append((i, chunk_file))
                    
                    del rgb_chunk
                    chunk_pbar.update(1)
                    
                    if (j + 1) % 10 == 0:
                        gc.collect()
            
            # Clean up workers for this strip
            for i in range(n_process):
                inqueue.put('STOP')
            for p in workers:
                p.join()
            feedp.join(1.)
            
            # Build strip from chunks (using mmap to avoid loading everything)
            strip_buffer = np.zeros((strip_height, nx, 3), dtype=np.uint8)
            
            for i in range(nc):
                chunk_file = os.path.join(tiles_dir, f'temp_chunk_{i:04d}.npy')
                # Memory-map the chunk file to avoid loading entire chunk
                chunk_mmap = np.load(chunk_file, mmap_mode='r')
                strip_buffer[:, bx[i]:ex[i], :] = chunk_mmap[:, :, :]
                del chunk_mmap
            
            # Flip vertically using C++ extension
            strip_buffer = image_processor.flip_vertical(strip_buffer)
            
            # Split into tiles using C++ extension
            tiles = image_processor.split_into_tiles(strip_buffer, TILE_SIZE)
            del strip_buffer
            
            # Save tiles for this strip
            tile_row = strips_per_height - 1 - strip_idx
            for tile_x_idx, tile_data in enumerate(tiles):
                save_tile(tile_data, max_level, tile_x_idx, tile_row, tiles_dir)
            
            del tiles
            
            # Delete temporary chunk files for this strip
            for i in range(nc):
                chunk_file = os.path.join(tiles_dir, f'temp_chunk_{i:04d}.npy')
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            
            gc.collect()
            strip_pbar.update(1)
    
    _log.info('All max-resolution tiles created')
    
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
        
        
