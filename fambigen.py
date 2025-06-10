# Ambigram Generation Script: Multi-Strategy Version
#
# This script experiments with different alignment strategies to generate
# ambigram glyphs from character pairs.
#
# Required Libraries:
# pip install fonttools skia-pathops svgwrite numpy Pillow scikit-image
#
import svgwrite
import math
import os
import io
import numpy as np
import traceback
import string
import argparse
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize, binary_opening, binary_closing, binary_dilation
from skimage.measure import find_contours, approximate_polygon
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontPens.flattenPen import FlattenPen
from fontTools.misc.transform import Transform
from pathops import Path as SkiaPath, union, difference, xor
from scipy.ndimage import rotate
from cairosvg import svg2png

import faulthandler

# --- Pen for Path Extraction ---

class SkiaPathPen(BasePen):
    """A pen to convert glyph outlines into a skia-pathops Path object."""
    def __init__(self, glyphSet=None):
        super().__init__(glyphSet)
        self.path = SkiaPath()

    def _moveTo(self, p):
        self.path.moveTo(p[0], p[1])

    def _lineTo(self, p):
        self.path.lineTo(p[0], p[1])

    def _curveToOne(self, p1, p2, p3):
        self.path.cubicTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

    def _qCurveToOne(self, p1, p2):
        self.path.quadTo(p1[0], p1[1], p2[0], p2[1])

    def _closePath(self):
        self.path.close()

# --- Strategy Functions ---
# These functions now return a single, final merged path object.

def align_using_centroid(path1_raw, path2_rotated, glyph_set, pair=""):
    """Aligns two paths by their geometric centroids and returns the union."""
    print("  -> Using Centroid Alignment")
    
    bounds1 = path1_raw.bounds
    bounds2 = path2_rotated.bounds
    if not bounds1 or not bounds2: return None
        
    cx1 = (bounds1[0] + bounds1[2]) / 2
    cy1 = (bounds1[1] + bounds1[3]) / 2
    cx2 = (bounds2[0] + bounds2[2]) / 2
    cy2 = (bounds2[1] + bounds2[3]) / 2

    transform1 = Transform().translate(-cx1, -cy1)
    transform2 = Transform().translate(-cx2, -cy2)

    pen1_aligned = SkiaPathPen(glyph_set)
    pen2_aligned = SkiaPathPen(glyph_set)
    path1_raw.draw(TransformPen(pen1_aligned, transform1))
    path2_rotated.draw(TransformPen(pen2_aligned, transform2))
    
    result_pen = SkiaPathPen(glyph_set)
    union([pen1_aligned.path, pen2_aligned.path], result_pen)
    return result_pen.path

def align_using_principal_axis(path1_raw, path2_rotated, glyph_set, pair=""):
    """Aligns two paths by their principal axes and returns the union."""
    print("  -> Using Principal Axis Alignment")

    def get_axis_transform_from_path(path):
        points = [p for verb, pts in path for p in pts]
        if len(points) < 2: return Transform()
        
        bounds = path.bounds
        if not bounds: return Transform()
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2

        coords = np.array(points)
        pca_center = np.mean(coords, axis=0) 
        cov = np.cov(coords - pca_center, rowvar=False)
        _, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, -1]
        angle = np.arctan2(principal_axis[1], principal_axis[0])
        
        return Transform().translate(-cx, -cy).rotate(-angle)

    try:
        transform1 = get_axis_transform_from_path(path1_raw)
        transform2 = get_axis_transform_from_path(path2_rotated)

        pen1_aligned = SkiaPathPen(glyph_set)
        pen2_aligned = SkiaPathPen(glyph_set)
        path1_raw.draw(TransformPen(pen1_aligned, transform1))
        path2_rotated.draw(TransformPen(pen2_aligned, transform2))
        
        result_pen = SkiaPathPen(glyph_set)
        union([pen1_aligned.path, pen2_aligned.path], result_pen)
        return result_pen.path
    except (np.linalg.LinAlgError, ValueError):
        print("  -> Warning: Could not compute principal axis. Falling back to centroid.")
        return align_using_centroid(path1_raw, path2_rotated, glyph_set)

# --- Make sure all these are imported at the top of your file ---
# from skimage.morphology import skeletonize, binary_dilation
# from scipy.ndimage import rotate
# from skimage.measure import find_contours, approximate_polygon

def get_vector_skeleton(char, font, glyph_set):
    """
    Takes a single character and returns a clean, simplified, vector-based skeleton path.
    """
    # 1. Get the path for the single character.
    glyph_name = font.getBestCmap().get(ord(char))
    if not glyph_name:
        return None
    
    char_pen = SkiaPathPen(glyph_set)
    glyph_set[glyph_name].draw(char_pen)
    char_path = char_pen.path
    if not char_path.bounds:
        return None

    # 2. Rasterize the single character into a filled shape.
    img_size = 256
    padding = 20
    bounds = char_path.bounds
    transform = Transform().translate(padding, padding).scale((img_size - 2 * padding) / max(bounds[2] - bounds[0], bounds[3] - bounds[1])).translate(-bounds[0], -bounds[1])
    
    render_pen = SkiaPathPen(glyph_set)
    char_path.draw(TransformPen(render_pen, transform))

    img = Image.new("1", (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)

    contours, current_contour = [], []
    for verb, pts in render_pen.path:
        if verb == "moveTo":
            if current_contour: contours.append(current_contour)
            current_contour = list(pts)
        else:
            current_contour.extend(pts)
    if current_contour: contours.append(current_contour)

    for contour in contours:
        flat_contour = [coord for point in contour for coord in point]
        if len(flat_contour) >= 6:
             draw.polygon(flat_contour, fill=1)
    
    # 3. Create a pixel skeleton. Because the input is a clean character, the skeleton will be a clean line.
    pixel_skeleton = skeletonize(np.array(img).astype(np.uint8), method='lee')

    # 4. Vectorize and simplify the pixel skeleton.
    skeleton_contours = find_contours(pixel_skeleton, 0.5)
    if not skeleton_contours:
        return None
        
    simplified_skeleton_points = []
    for contour in skeleton_contours:
        simplified_contour = approximate_polygon(contour, tolerance=1.5)
        simplified_skeleton_points.extend(simplified_contour[:-1])
    
    if not simplified_skeleton_points:
        return None

    # 5. Build the final vector skeleton path and return it.
    skeleton_pen = SkiaPathPen(glyph_set)
    # The points are in image space, so we use the inverse transform to get them back to font space.
    inverse_transform = transform.inverse()
    
    # We create a single path, using moveTo for the start of each disconnected contour piece.
    # This assumes find_contours might return multiple pieces for some skeletons (e.g., the letter 'x').
    for contour in skeleton_contours:
        # Move to the start of the contour piece
        y, x = contour[0]
        px, py = inverse_transform.transformPoint((x,y))
        skeleton_pen.moveTo((px,py))
        # Line to the rest of the points
        for y, x in contour[1:]:
            px, py = inverse_transform.transformPoint((x,y))
            skeleton_pen.lineTo((px,py))

    return skeleton_pen.path


def rasterize_path(draw_context, path_to_draw, line_width=2):
    """A robust helper to draw a complex SkiaPath onto a Pillow ImageDraw context."""
    if not path_to_draw or not path_to_draw.bounds:
        return

    # This contour-building logic is the most stable method we've found.
    contours = []
    current_contour = []
    for verb, pts in path_to_draw:
        if verb == "moveTo":
            if current_contour: contours.append(current_contour)
            current_contour = list(pts)
        elif verb == "closePath":
            if current_contour:
                contours.append(current_contour)
            current_contour = []
        else: # lineTo, quadTo, cubicTo all add points
            current_contour.extend(pts)
    if current_contour:
        contours.append(current_contour)

    # Now draw each collected contour as a line
    for contour in contours:
        if len(contour) > 1:
            # Flatten the list of (x,y) tuples for Pillow's line method
            flat_line = [coord for point in contour for coord in point]
            draw_context.line(flat_line, fill=1, width=line_width)

def generate_using_outline(path1_raw, path2_rotated, glyph_set, pair=""):
    """
    Creates a calligraphic 'outline' style glyph using purely vector operations.
    This version removes the failing .simplify() call and returns the direct
    result of the boolean XOR operation.
    """
    try:
        # Step 1: Get the base merged shape.
        base_merged_path = align_using_centroid(path1_raw, path2_rotated, glyph_set, pair)
        if not base_merged_path or not base_merged_path.bounds:
            return None

        # Step 2: Calculate the center of the shape for scaling.
        bounds = base_merged_path.bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2

        # Step 3: Create a scaled-down version of the path.
        scale_factor = 0.88
        
        scale_down_transform = Transform().translate(-cx, -cy).scale(scale_factor).translate(cx, cy)

        scaled_down_pen = SkiaPathPen(glyph_set)
        base_merged_path.draw(TransformPen(scaled_down_pen, scale_down_transform))
        scaled_down_path = scaled_down_pen.path

        # Step 4: Use the explicit XOR function to create the outline.
        outline_pen = SkiaPathPen(glyph_set)
        xor((base_merged_path,), (scaled_down_path,), outline_pen)
        outline_path = outline_pen.path
        
        # Step 5: Return the result directly, skipping the failing simplify() step.
        if not outline_path or not outline_path.bounds:
            return None

        return outline_path

    except Exception as e:
        print(f"  -> ERROR in outline strategy: {e}")
        traceback.print_exc()
        return None
        

def generate_using_centerline_trace(path1_raw, path2_rotated, glyph_set, pair=""):
    """
    Creates an ambigram by skeletonizing each character, aligning them by their
    centroids, rasterizing them, and then stroking the result.
    """
    try:
        font = glyph_set.font
        char1, char2 = pair[0], pair[1]

        # Step 1: Get a clean vector skeleton for each character.
        skeleton1 = get_vector_skeleton(char1, font, glyph_set)
        skeleton2 = get_vector_skeleton(char2, font, glyph_set)
        if not skeleton1 or not skeleton2: return None

        # Step 2: Rotate skeleton2.
        pen2_rotated = SkiaPathPen(glyph_set)
        skeleton2.draw(TransformPen(pen2_rotated, Transform().rotate(math.pi)))
        skeleton2_rotated = pen2_rotated.path

        # Step 3: Individually align each skeleton's centroid to the origin (0,0).
        bounds1 = skeleton1.bounds
        cx1 = (bounds1[0] + bounds1[2]) / 2
        cy1 = (bounds1[1] + bounds1[3]) / 2
        transform1 = Transform().translate(-cx1, -cy1)
        
        pen1_aligned = SkiaPathPen(glyph_set)
        skeleton1.draw(TransformPen(pen1_aligned, transform1))
        aligned_path1 = pen1_aligned.path

        bounds2 = skeleton2_rotated.bounds
        cx2 = (bounds2[0] + bounds2[2]) / 2
        cy2 = (bounds2[1] + bounds2[3]) / 2
        transform2 = Transform().translate(-cx2, -cy2)

        pen2_aligned = SkiaPathPen(glyph_set)
        skeleton2_rotated.draw(TransformPen(pen2_aligned, transform2))
        aligned_path2 = pen2_aligned.path

        # Step 4: Now that both are centered, find their combined bounds
        # and create a single transform to fit them onto the canvas.
        union_pen = SkiaPathPen(glyph_set)
        union((aligned_path1, aligned_path2), union_pen)
        combined_path = union_pen.path
        if not combined_path or not combined_path.bounds: return None

        img_size = 256
        padding = 20
        bounds = combined_path.bounds
        transform_to_image = Transform().translate(padding, padding).scale((img_size - 2 * padding) / max(bounds[2] - bounds[0], bounds[3] - bounds[1])).translate(-bounds[0], -bounds[1])

        # Step 5: Draw both of the individually-aligned paths onto the same canvas.
        img = Image.new("1", (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)

        for path in [aligned_path1, aligned_path2]:
            render_pen = SkiaPathPen(glyph_set)
            path.draw(TransformPen(render_pen, transform_to_image))
            rasterize_path(draw, render_pen.path, line_width=2)
            
        # --- MISSING BLOCK RESTORED HERE ---
        # Step 6: Create the calligraphic pen kernel.
        max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        scale = (img_size - 2 * padding) / max_dim if max_dim > 0 else 1
        # We define a single size for our pen canvas and thickness for the line
        pen_size = int(max(3, 10 * scale))
        pen_thickness = int(max(1, 3 * scale)) # Control thickness here

        # Create a square canvas
        chisel_kernel = np.zeros((pen_size, pen_size), dtype=np.uint8)
        
        # Draw a vertical line down the center
        center = pen_size // 2
        p_thick_half = pen_thickness // 2
        chisel_kernel[:, center - p_thick_half : center + p_thick_half + 1] = 1
        
        # Rotate the canvas with the vertical line 45 degrees clockwise
        # Note: In scipy.ndimage.rotate, a negative angle is clockwise
        chisel_kernel = rotate(chisel_kernel, -45, reshape=False, order=0)        
        
        # Step 7: Dilate the rasterized skeleton with the calligraphic kernel.
        final_bitmap = binary_dilation(np.array(img), footprint=chisel_kernel)
        
        # Step 8: Vectorize the final bitmap back to a path.
        found_contours = find_contours(final_bitmap.astype(np.uint8), level=0.5)
        if not found_contours: return None

        main_contour = max(found_contours, key=len)
        hole_contours = [c for c in found_contours if c is not main_contour and len(c) > 20]
        
        final_path_pen = SkiaPathPen(glyph_set)
        inverse_transform = transform_to_image.inverse()
        
        y, x = main_contour[0]
        px, py = inverse_transform.transformPoint((x, y))
        final_path_pen.moveTo((px, py))
        for y, x in main_contour[1:]:
            px, py = inverse_transform.transformPoint((x, y))
            final_path_pen.lineTo((px, py))
        final_path_pen.closePath()

        for hole in hole_contours:
            reversed_hole = hole[::-1]
            y, x = reversed_hole[0]
            px, py = inverse_transform.transformPoint((x, y))
            final_path_pen.moveTo((px, py))
            for y, x in reversed_hole[1:]:
                px, py = inverse_transform.transformPoint((x, y))
                final_path_pen.lineTo((px, py))
            final_path_pen.closePath()

        return final_path_pen.path

    except Exception as e:
        print(f"  -> ERROR in centerline trace: {e}")
        traceback.print_exc()
        return None

# --- Core Generation Logic ---

def generate_ambigram_svg(font, pair, output_dir, strategy_func):
    """Generates an ambigram SVG using a specified strategy function."""
    if len(pair) != 2: return
        
    char1, char2 = pair[0], pair[1]
    
    strategy_name = strategy_func.__name__.replace('align_using_', '').replace('generate_using_', '')
    strategy_output_dir = os.path.join(output_dir, f"generated_glyphs_{strategy_name}")
    if not os.path.exists(strategy_output_dir):
        os.makedirs(strategy_output_dir)
        
    output_filename = os.path.join(strategy_output_dir, f"{pair}.svg")

    glyph_set = font.getGlyphSet()
    glyph_name1 = font.getBestCmap().get(ord(char1))
    glyph_name2 = font.getBestCmap().get(ord(char2))

    if not glyph_name1 or not glyph_name2: return

    # 1. Extract raw and rotated paths
    pen1_raw = SkiaPathPen(glyph_set)
    glyph_set[glyph_name1].draw(pen1_raw)
    path1_raw = pen1_raw.path

    pen2_rotated = SkiaPathPen(glyph_set)
    glyph_set[glyph_name2].draw(TransformPen(pen2_rotated, Transform().rotate(math.pi)))
    path2_rotated = pen2_rotated.path

    if not path1_raw.bounds or not path2_rotated.bounds: return

    # 2. Get the final merged path from the strategy function
    #merged_skia_path = strategy_func(path1_raw, path2_rotated, glyph_set)
    merged_skia_path = strategy_func(path1_raw, path2_rotated, glyph_set, pair)

    if not merged_skia_path:
        print(f"  -> Warning: Strategy '{strategy_name}' failed for '{pair}'. Skipping.")
        return

    # 3. Create and save the SVG file
    bounds = merged_skia_path.bounds
    if not bounds or (bounds[0] == 0 and bounds[1] == 0 and bounds[2] == 0 and bounds[3] == 0): return

    padding = 50
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    
    svg_pen = SVGPathPen(glyph_set)
    merged_skia_path.draw(svg_pen)
    svg_path_data = svg_pen.getCommands()

    if not svg_path_data: return

    viewbox_str = f"{left - padding} {top - padding} {width + padding*2} {height + padding*2}"
    
    dwg = svgwrite.Drawing(output_filename, profile='tiny', viewBox=viewbox_str)
    g = dwg.g(transform=f"translate(0, {bottom + top}) scale(1, -1)")
    g.add(dwg.path(d=svg_path_data, fill='black'))
    dwg.add(g)
    dwg.save()
    print(f"  -> Saved to {output_filename}")

def create_ambigram_from_string(word1, strategy_name, output_filename, word2=None):
    """
    Creates a single composite ambigram image from one or two words,
    preserving case.

    Args:
        word1 (str): The word that reads forwards.
        strategy_name (str): The name of the generation strategy (e.g., "outline").
        output_filename (str): The filename for the final PNG image.
        word2 (str, optional): The word that reads upon rotation.
                               If None, a palindromic ambigram of word1 is created.
    """
    # If word2 is not provided, the reversed word1 is used.
    if not word2:
        word2 = word1[::-1]
    
    print(f"\n--- Composing ambigram for '{word1}' / '{word2}' ---")
    
    # Define the directory where the glyphs are stored
    glyph_dir = os.path.join(".", f"generated_glyphs_{strategy_name}")

    # Determine the required SVG file for each letter position, preserving case.
    required_files = [f"{c1}{c2}.svg" for c1, c2 in zip(word1, word2)]
    
    glyph_images = []
    for filename in required_files:
        filepath = os.path.join(glyph_dir, filename)
        if not os.path.exists(filepath):
            print(f"  -> Warning: Required glyph file not found, skipping: {filepath}")
            continue
        
        try:
            png_data = svg2png(url=filepath)
            glyph_image = Image.open(io.BytesIO(png_data))
            glyph_images.append(glyph_image)
            print(f"  -> Loaded and rendered {filename}")
        except Exception as e:
            print(f"  -> ERROR: Could not process {filename}. Details: {e}")

    if not glyph_images:
        print("Could not render any glyphs. Aborting composition.")
        return

    # Calculate dimensions for the final composite image
    total_width = sum(img.width for img in glyph_images)
    max_height = max(img.height for img in glyph_images)

    # Create a new blank canvas with a white background
    composite_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 255))
    
    current_x = 0
    for img in glyph_images:
        composite_image.paste(img, (current_x, 0), img)
        current_x += img.width

    composite_image.save(output_filename)
    print(f"Ambigram saved successfully to {output_filename}")

# --- Main Execution ---

if __name__ == "__oldmain__":
    faulthandler.enable() 
    #FONT_FILE_PATH = "C:/temp/roboto/static/Roboto-Condensed-Thin.ttf"
    FONT_FILE_PATH = "C:/Windows/Fonts/Arial.ttf"
    #PAIRS_TO_GENERATE = ['nu', 'zs', 'ab', 'hi', 'do', 'bp', 'mw', 'sx', 'bd', 'aa', 'db']
    PAIRS_TO_GENERATE = [c1 + c2 for c1 in string.ascii_lowercase for c2 in string.ascii_lowercase]
    
    STRATEGY_FUNCTIONS = [
        align_using_centroid,
        align_using_principal_axis,
        generate_using_outline,
    ]
    
    if not os.path.exists(FONT_FILE_PATH):
        print(f"ERROR: Font file not found at '{FONT_FILE_PATH}'")
    else:
        print(f"Loading font from {FONT_FILE_PATH}...")
        try:
            font = TTFont(FONT_FILE_PATH)
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load font. Aborting. Details: {e}")
            exit()

        for strategy_func in STRATEGY_FUNCTIONS:
            print(f"\n--- Applying Strategy: {strategy_func.__name__} ---")
            for pair in PAIRS_TO_GENERATE:
                print(f"\n- Generating pair: '{pair}'")
                generate_ambigram_svg(font, pair, ".", strategy_func)

if __name__ == "__main__":
    # --- Step 1: Set up the command-line argument parser ---
    parser = argparse.ArgumentParser(
        description="Generate a composite ambigram from one or two words using a specified font.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "word1", 
        type=str, 
        help="The first word to create the ambigram from (reads forwards)."
    )
    parser.add_argument(
        "word2", 
        type=str, 
        nargs='?', # Makes this argument optional
        default=None, 
        help="(Optional) The second word (reads when rotated 180 degrees).\nIf omitted, a palindromic ambigram of word1 will be created."
    )
    parser.add_argument(
        "-f", "--font", 
        type=str, 
        default="C:/Windows/Fonts/Arial.ttf",
        help="Path to the TTF font file to use.\nDefaults to Arial on Windows."
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default="outline",
        choices=['centroid', 'principal_axis', 'outline', 'centerline_trace'],
        help="The generation strategy to use. Defaults to 'outline'."
    )
    
    args = parser.parse_args()

    # --- Step 2: Assign parsed arguments to our variables ---
    INPUT_WORD = args.word1
    INPUT_WORD2 = args.word2
    FONT_FILE_PATH = args.font
    
    strategy_map = {
        'centroid': align_using_centroid,
        'principal_axis': align_using_principal_axis,
        'outline': generate_using_outline,
        'centerline_trace': generate_using_centerline_trace
    }
    STRATEGY_TO_USE = strategy_map[args.strategy]

    # --- The rest of the script now uses the arguments from the command line ---

    # Input Validation
    if INPUT_WORD2 and len(INPUT_WORD) != len(INPUT_WORD2):
        print(f"ERROR: Input words '{INPUT_WORD}' and '{INPUT_WORD2}' must be the same length.")
        exit()

    if not os.path.exists(FONT_FILE_PATH):
        print(f"ERROR: Font file not found at '{FONT_FILE_PATH}'")
        exit()

    # Stage 1: Generate necessary glyphs
    strategy_name = STRATEGY_TO_USE.__name__.replace('generate_using_', '')
    print(f"--- Using strategy: {strategy_name} ---")

    comparison_word = INPUT_WORD2[::-1] if INPUT_WORD2 else INPUT_WORD[::-1]
    pairs_to_generate = sorted(list(set([c1 + c2 for c1, c2 in zip(INPUT_WORD, comparison_word)])))
    
    print(f"Required pairs to generate: {pairs_to_generate}")

    try:
        font = TTFont(FONT_FILE_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load font. Aborting. Details: {e}")
        exit()

    for pair in pairs_to_generate:
        print(f"\n- Generating glyph for pair: '{pair}'")
        generate_ambigram_svg(font, pair, ".", STRATEGY_TO_USE)

    # Stage 2: Compose the final image
    output_filename = f"{INPUT_WORD}{'-' + INPUT_WORD2 if INPUT_WORD2 else ''}_ambigram.png"
    create_ambigram_from_string(INPUT_WORD, strategy_name, output_filename, word2=INPUT_WORD2[::-1])