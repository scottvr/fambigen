# fambigen.py 
#
# This script generates ambigram glyphs from character pairs (taken from a specified TTF i
# or WOFF) using various strategies and can compose them into a single image based on 
# input words.

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
from fontTools.pens.basePen import BasePen, NullPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.misc.transform import Transform
from fontTools.pens.recordingPen import RecordingPen
from fontPens.flattenPen import FlattenPen
from pathops import Path as SkiaPath, union, difference, xor, intersection
from scipy.ndimage import rotate
from scipy.spatial.distance import directed_hausdorff
from cairosvg import svg2png

#import faulthandler

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

def create_rect_path(bounds_tuple):
    """Creates a rectangular SkiaPath from a (left, top, right, bottom) tuple."""
    left, top, right, bottom = bounds_tuple
    rect_path = SkiaPath()
    rect_path.moveTo(left, top)
    rect_path.lineTo(right, top)
    rect_path.lineTo(right, bottom)
    rect_path.lineTo(left, bottom)
    rect_path.close()
    return rect_path


def convert_path_to_points(path, num_points=150):
    """Converts a SkiaPath to a numpy array of 'num_points' equidistant points."""
    if not path or not path.bounds:
        return np.array([])

    recording_pen = RecordingPen()
    flatten_pen = FlattenPen(recording_pen, approximateSegmentLength=5)
    path.draw(flatten_pen)

    pts = []
    for command, data in recording_pen.value:
        if data:
            pts.append(data[-1])
    
    if len(pts) < 2:
        return np.array([pts]) if pts else np.array([])

    pts = np.array(pts)
    segments = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    perimeter = np.sum(segments)
    
    if perimeter == 0:
        return pts # Return the existing points if path has no length

    cumulative_dist = np.insert(np.cumsum(segments), 0, 0)
    
    dist_samples = np.linspace(0, perimeter, num_points)
    
    interp_pts = np.zeros((num_points, 2))
    for i, d in enumerate(dist_samples):
        # Find which segment this distance falls into
        segment_index = np.searchsorted(cumulative_dist, d, side='right') - 1
        segment_index = max(0, min(segment_index, len(segments) - 1))

        dist_into_segment = d - cumulative_dist[segment_index]
        segment_len = segments[segment_index]
        
        if segment_len > 0:
            ratio = dist_into_segment / segment_len
            start_pt = pts[segment_index]
            end_pt = pts[segment_index + 1]
            interp_pts[i] = start_pt + ratio * (end_pt - start_pt)
        else:
            # If segment has zero length, just use the start point
            interp_pts[i] = pts[segment_index]

    return interp_pts

def normalize_point_cloud(points):
    """Translates and scales a point cloud for consistent comparison."""
    if points.shape[0] < 2:
        return points
    
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    max_val = np.max(np.abs(centered_points))
    if max_val > 0:
        normalized_points = centered_points / max_val
    else:
        normalized_points = centered_points
        
    return normalized_points

def calculate_legibility_score(generated_path, canonical_path1, canonical_path2):
    """
    Calculates a legibility score for a generated ambigram path by comparing it
    to the two source paths using the Hausdorff distance.
    
    A lower score is better.
    """
    if not generated_path or not canonical_path1 or not canonical_path2:
        return float('inf') # Return a worst-case score for invalid paths

    rotation = Transform().rotate(math.pi)
    rotated_pen = SkiaPathPen()
    generated_path.draw(TransformPen(rotated_pen, rotation))
    generated_path_rotated = rotated_pen.path

    points_gen_upright = convert_path_to_points(generated_path)
    points_gen_rotated = convert_path_to_points(generated_path_rotated)
    points_canon1 = convert_path_to_points(canonical_path1)
    points_canon2 = convert_path_to_points(canonical_path2)

    if any(p.size == 0 for p in [points_gen_upright, points_gen_rotated, points_canon1, points_canon2]):
        return float('inf') # Cannot compare if any path is empty

    norm_gen_upright = normalize_point_cloud(points_gen_upright)
    norm_gen_rotated = normalize_point_cloud(points_gen_rotated)
    norm_canon1 = normalize_point_cloud(points_canon1)
    norm_canon2 = normalize_point_cloud(points_canon2)

    # The full Hausdorff distance is the max of the two directed distances.
    # This measures how far P1 is from P2 and vice-versa.
    d1_forward = directed_hausdorff(norm_gen_upright, norm_canon1)[0]
    d1_backward = directed_hausdorff(norm_canon1, norm_gen_upright)[0]
    d1 = max(d1_forward, d1_backward)

    d2_forward = directed_hausdorff(norm_gen_rotated, norm_canon2)[0]
    d2_backward = directed_hausdorff(norm_canon2, norm_gen_rotated)[0]
    d2 = max(d2_forward, d2_backward)

    rms_score = np.sqrt((d1**2 + d2**2) / 2)
    
    return rms_score

def align_using_centroid(path1_raw, path2_rotated, pair=""):
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

    pen1_aligned = SkiaPathPen()
    pen2_aligned = SkiaPathPen()
    path1_raw.draw(TransformPen(pen1_aligned, transform1))
    path2_rotated.draw(TransformPen(pen2_aligned, transform2))
    
    result_pen = SkiaPathPen()
    union([pen1_aligned.path, pen2_aligned.path], result_pen)
    return result_pen.path

def align_using_principal_axis(path1_raw, path2_rotated, pair=""):
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

        pen1_aligned = SkiaPathPen()
        pen2_aligned = SkiaPathPen()
        path1_raw.draw(TransformPen(pen1_aligned, transform1))
        path2_rotated.draw(TransformPen(pen2_aligned, transform2))
        
        result_pen = SkiaPathPen()
        union([pen1_aligned.path, pen2_aligned.path], result_pen)
        return result_pen.path
    except (np.linalg.LinAlgError, ValueError):
        print("  -> Warning: Could not compute principal axis. Falling back to centroid.")
        return align_using_centroid(path1_raw, path2_rotated, glyph_set)

def calculate_path_area(path, canvas_size=256):
    """Calculates the approximate area of a SkiaPath by rasterizing and filling it."""
    if not path or not path.bounds:
        return 0
    
    bounds = path.bounds
    # Prevent division by zero for paths with no width or height
    path_width = bounds[2] - bounds[0]
    path_height = bounds[3] - bounds[1]
    if path_width <= 0 or path_height <= 0:
        return 0

    # Scale the path to fit within the canvas for consistent area measurement
    padding = canvas_size * 0.05 # 5% padding
    usable_size = canvas_size - 2 * padding
    scale = usable_size / max(path_width, path_height)
    transform = Transform().translate(-bounds[0], -bounds[1]).translate(padding, padding).scale(scale)
    
    render_pen = SkiaPathPen()
    path.draw(TransformPen(render_pen, transform))
    
    img = Image.new("1", (canvas_size, canvas_size), 0)
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
        if len(flat_contour) >= 6: # Need at least 3 points for a polygon
             draw.polygon(flat_contour, fill=1)

    # The "area" is the sum of all white pixels
    return np.sum(np.array(img))

def align_using_iterative_registration(path1_raw, path2_rotated, pair=""):
    """
    Aligns two paths by iteratively searching for the translation that
    maximizes the area of their intersection, then returns their union.
    """
    print("  -> Using Iterative Shape Registration")

    bounds1 = path1_raw.bounds
    bounds2 = path2_rotated.bounds
    if not bounds1 or not bounds2: return None
        
    cx1, cy1 = (bounds1[0] + bounds1[2]) / 2, (bounds1[1] + bounds1[3]) / 2
    center_transform1 = Transform().translate(-cx1, -cy1)
    pen1_centered = SkiaPathPen()
    path1_raw.draw(TransformPen(pen1_centered, center_transform1))
    path1_centered = pen1_centered.path

    best_overlap = -1.0
    best_transform_for_path2 = None
    cx2, cy2 = (bounds2[0] + bounds2[2]) / 2, (bounds2[1] + bounds2[3]) / 2

    # Define search parameters based on the glyph's size
    max_dim = max(bounds1[2] - bounds1[0], bounds1[3] - bounds1[1], bounds2[2] - bounds2[0], bounds2[3] - bounds2[1])
    search_range = int(max_dim * 0.1)
    num_steps_in_radius = 20 
    step = max(2, int(search_range / num_steps_in_radius))

    print(f"  -> Searching in a {search_range*2}x{search_range*2} unit area with a step of {step}...")
    for dx in range(-search_range, search_range + 1, step):
        for dy in range(-search_range, search_range + 1, step):
            # The current transform for path2 is:
            # 1. Move its original centroid to the origin (like path1)
            # 2. Apply the iterative offset (dx, dy) to search for a better fit
            current_transform = Transform().translate(-cx2 + dx, -cy2 + dy)
            
            path2_temp_pen = SkiaPathPen()
            path2_rotated.draw(TransformPen(path2_temp_pen, current_transform))
            path2_transformed = path2_temp_pen.path

            intersection_pen = SkiaPathPen()
            intersection((path1_centered,), (path2_transformed,), intersection_pen)
            overlap_area = calculate_path_area(intersection_pen.path)

            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_transform_for_path2 = current_transform
    
    if best_transform_for_path2 is None:
        print("  -> Warning: Iterative registration failed to find an overlap. Falling back to centroid.")
        return align_using_centroid(path1_raw, path2_rotated, pair)

    print(f"  -> Best overlap found with score: {best_overlap:.4f}")

    # Apply the best found transform to path2
    pen2_final_aligned = SkiaPathPen()
    path2_rotated.draw(TransformPen(pen2_final_aligned, best_transform_for_path2))

    # Union the centered path1 and the optimally aligned path2
    result_pen = SkiaPathPen()
    union([path1_centered, pen2_final_aligned.path], result_pen)
    return result_pen.path


def get_vector_skeleton(char, font):
    """
    Takes a single character and returns a clean, simplified, vector-based skeleton path.
    """
    glyph_name = font.getBestCmap().get(ord(char))
    if not glyph_name:
        return None
    
    char_pen = SkiaPathPen()
    glyph_set[glyph_name].draw(char_pen)
    char_path = char_pen.path
    if not char_path.bounds:
        return None

    img_size = 256
    padding = 20
    bounds = char_path.bounds
    transform = Transform().translate(padding, padding).scale((img_size - 2 * padding) / max(bounds[2] - bounds[0], bounds[3] - bounds[1])).translate(-bounds[0], -bounds[1])
    
    render_pen = SkiaPathPen()
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
    
    pixel_skeleton = skeletonize(np.array(img).astype(np.uint8), method='lee')

    skeleton_contours = find_contours(pixel_skeleton, 0.5)
    if not skeleton_contours:
        return None
        
    simplified_skeleton_points = []
    for contour in skeleton_contours:
        simplified_contour = approximate_polygon(contour, tolerance=1.5)
        simplified_skeleton_points.extend(simplified_contour[:-1])
    
    if not simplified_skeleton_points:
        return None

    skeleton_pen = SkiaPathPen()
    inverse_transform = transform.inverse()
    
    for contour in skeleton_contours:
        y, x = contour[0]
        px, py = inverse_transform.transformPoint((x,y))
        skeleton_pen.moveTo((px,py))
        for y, x in contour[1:]:
            px, py = inverse_transform.transformPoint((x,y))
            skeleton_pen.lineTo((px,py))

    return skeleton_pen.path


def rasterize_path(draw_context, path_to_draw, line_width=2):
    """helper to draw a complex SkiaPath onto a Pillow ImageDraw context."""
    if not path_to_draw or not path_to_draw.bounds:
        return

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

    for contour in contours:
        if len(contour) > 1:
            flat_line = [coord for point in contour for coord in point]
            draw_context.line(flat_line, fill=1, width=line_width)

def generate_using_outline(path1_raw, path2_rotated, pair="", alignment_func=align_using_centroid):
    """
    Creates an outline glyph.
    """
    try:
        print(f"  -> Using '{alignment_func.__name__}' for alignment.")
        base_merged_path = alignment_func(path1_raw, path2_rotated, pair)

        if not base_merged_path or not base_merged_path.bounds:
            return None

        bounds = base_merged_path.bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2

        scale_factor = 0.88
        
        scale_down_transform = Transform().translate(-cx, -cy).scale(scale_factor).translate(cx, cy)

        scaled_down_pen = SkiaPathPen()
        base_merged_path.draw(TransformPen(scaled_down_pen, scale_down_transform))
        scaled_down_path = scaled_down_pen.path

        outline_pen = SkiaPathPen()
        xor((base_merged_path,), (scaled_down_path,), outline_pen)
        outline_path = outline_pen.path
        
        if not outline_path or not outline_path.bounds:
            return None

        return outline_path

    except Exception as e:
        print(f"  -> ERROR in outline strategy: {e}")
        traceback.print_exc()
        return None
        

def generate_using_centerline_trace(path1_raw, path2_rotated, pair=""):
    """
    Creates an ambigram by skeletonizing each character, aligning them by their
    centroids, rasterizing them, and then stroking the result.
    """
    try:
        font = glyph_set.font
        char1, char2 = pair[0], pair[1]

        skeleton1 = get_vector_skeleton(char1, font, glyph_set)
        skeleton2 = get_vector_skeleton(char2, font, glyph_set)
        if not skeleton1 or not skeleton2: return None

        pen2_rotated = SkiaPathPen()
        skeleton2.draw(TransformPen(pen2_rotated, Transform().rotate(math.pi)))
        skeleton2_rotated = pen2_rotated.path

        bounds1 = skeleton1.bounds
        cx1 = (bounds1[0] + bounds1[2]) / 2
        cy1 = (bounds1[1] + bounds1[3]) / 2
        transform1 = Transform().translate(-cx1, -cy1)
        
        pen1_aligned = SkiaPathPen()
        skeleton1.draw(TransformPen(pen1_aligned, transform1))
        aligned_path1 = pen1_aligned.path

        bounds2 = skeleton2_rotated.bounds
        cx2 = (bounds2[0] + bounds2[2]) / 2
        cy2 = (bounds2[1] + bounds2[3]) / 2
        transform2 = Transform().translate(-cx2, -cy2)

        pen2_aligned = SkiaPathPen()
        skeleton2_rotated.draw(TransformPen(pen2_aligned, transform2))
        aligned_path2 = pen2_aligned.path

        union_pen = SkiaPathPen()
        union((aligned_path1, aligned_path2), union_pen)
        combined_path = union_pen.path
        if not combined_path or not combined_path.bounds: return None

        img_size = 256
        padding = 20
        bounds = combined_path.bounds
        transform_to_image = Transform().translate(padding, padding).scale((img_size - 2 * padding) / max(bounds[2] - bounds[0], bounds[3] - bounds[1])).translate(-bounds[0], -bounds[1])

        img = Image.new("1", (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)

        for path in [aligned_path1, aligned_path2]:
            render_pen = SkiaPathPen(glyph_set)
            path.draw(TransformPen(render_pen, transform_to_image))
            rasterize_path(draw, render_pen.path, line_width=2)
            
        max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        scale = (img_size - 2 * padding) / max_dim if max_dim > 0 else 1
        pen_size = int(max(3, 10 * scale))
        pen_thickness = int(max(1, 3 * scale)) # Control thickness here

        chisel_kernel = np.zeros((pen_size, pen_size), dtype=np.uint8)
        
        center = pen_size // 2
        p_thick_half = pen_thickness // 2
        chisel_kernel[:, center - p_thick_half : center + p_thick_half + 1] = 1
        chisel_kernel = rotate(chisel_kernel, -45, reshape=False, order=0)        
        
        final_bitmap = binary_dilation(np.array(img), footprint=chisel_kernel)
        
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

def generate_using_half_letters(path1_raw, path2_rotated, pair=""):
    """
    Creates an ambigram by clipping the TOP half of each character FIRST,
    and then rotating and aligning the resulting pieces.
    """
    try:
        # Calculate alignment transforms from the FULL paths BEFORE clipping
        bounds1 = path1_raw.bounds
        if not bounds1: return None
        cx1, cy1 = (bounds1[0] + bounds1[2]) / 2, (bounds1[1] + bounds1[3]) / 2
        transform1 = Transform().translate(-cx1, -cy1)

        bounds2 = path2_rotated.bounds
        if not bounds2: return None
        cx2, cy2 = (bounds2[0] + bounds2[2]) / 2, (bounds2[1] + bounds2[3]) / 2
        transform2 = Transform().translate(-cx2, -cy2)

        # Get the top half of the original char1
        top_half_y_mid1 = (bounds1[1] + bounds1[3]) / 2
        clip_box1 = create_rect_path((bounds1[0] - 1, top_half_y_mid1, bounds1[2] + 1, bounds1[3] + 1))
        top_half_pen1 = SkiaPathPen()
        intersection((path1_raw,), (clip_box1,), top_half_pen1)
        top_half1 = top_half_pen1.path

        # Get the top half of the original char2
        path2_raw_temp_pen = SkiaPathPen()
        path2_rotated.draw(TransformPen(path2_raw_temp_pen, Transform().rotate(math.pi)))
        path2_raw_temp = path2_raw_temp_pen.path
        
        bounds2_raw = path2_raw_temp.bounds
        if not bounds2_raw: return None
        top_half_y_mid2 = (bounds2_raw[1] + bounds2_raw[3]) / 2
        clip_box2 = create_rect_path((bounds2_raw[0] - 1, top_half_y_mid2, bounds2_raw[2] + 1, bounds2_raw[3] + 1))
        top_half_pen2 = SkiaPathPen(glyph_set)
        intersection((path2_raw_temp,), (clip_box2,), top_half_pen2)
        top_half2 = top_half_pen2.path

        # Apply pre-calculated transforms to the clipped halves
        aligned_half1_pen = SkiaPathPen(glyph_set)
        top_half1.draw(TransformPen(aligned_half1_pen, transform1))
        aligned_half1 = aligned_half1_pen.path

        rotated_half2_pen = SkiaPathPen(glyph_set)
        top_half2.draw(TransformPen(rotated_half2_pen, Transform().rotate(math.pi)))
        
        aligned_half2_pen = SkiaPathPen(glyph_set)
        rotated_half2_pen.path.draw(TransformPen(aligned_half2_pen, transform2))
        aligned_half2 = aligned_half2_pen.path

        merged_halves_pen = SkiaPathPen(glyph_set)
        union((aligned_half1, aligned_half2), merged_halves_pen)
        merged_halves_path = merged_halves_pen.path
        if not merged_halves_path or not merged_halves_path.bounds: return None

        # Apply the vector outline logic
        bounds = merged_halves_path.bounds
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        scale_factor = 0.88
        
        scale_down_transform = Transform().translate(-cx, -cy).scale(scale_factor).translate(cx, cy)
        scaled_down_pen = SkiaPathPen(glyph_set)
        merged_halves_path.draw(TransformPen(scaled_down_pen, scale_down_transform))
        
        outline_pen = SkiaPathPen(glyph_set)
        xor((merged_halves_path,), (scaled_down_pen.path,), outline_pen)
        
        if not outline_pen.path or not outline_pen.path.bounds: return None
        return outline_pen.path

    except Exception as e:
        print(f"  -> ERROR in half_letters strategy: {e}\n{traceback.format_exc()}")
        return None
def save_path_as_svg(path_to_save, output_filename, glyph_set):
    """Saves a SkiaPath object to a specified SVG file."""
    if not path_to_save or not path_to_save.bounds:
        print(f"  -> Warning: Path for {os.path.basename(output_filename)} is empty. Skipping save.")
        return

    bounds = path_to_save.bounds
    padding = 50
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top

    svg_pen = SVGPathPen(glyph_set)
    path_to_save.draw(svg_pen)
    svg_path_data = svg_pen.getCommands()

    if not svg_path_data:
        print(f"  -> Warning: Could not get SVG command data for {os.path.basename(output_filename)}. Skipping save.")
        return

    viewbox_str = f"{left - padding} {top - padding} {width + padding*2} {height + padding*2}"
    
    dwg = svgwrite.Drawing(output_filename, profile='tiny', viewBox=viewbox_str)
    # The transform corrects for SVG's coordinate system (y-down) vs. font coordinates (y-up)
    g = dwg.g(transform=f"translate(0, {bottom + top}) scale(1, -1)")
    g.add(dwg.path(d=svg_path_data, fill='black'))
    dwg.add(g)
    dwg.save()
    print(f"  -> Saved to {output_filename}")

def generate_ambigram_svg(font1, font2, pair, output_dir, strategy_func, uniform_glyphs=False, alignment_func=align_using_centroid):
    """Generates an ambigram SVG using a specified strategy function."""
    if len(pair) != 2: return
        
    char1, char2 = pair[0], pair[1]
    
    align_name = alignment_func.__name__.replace('align_using_', '')
    strategy_name = strategy_func.__name__.replace('align_using_', '').replace('generate_using_', '')

    output_folder_name = f"generated_glyphs_{strategy_name}"
    if strategy_func == generate_using_outline: # Only add alignment name for relevant strategies
        output_folder_name += f"_{align_name}"

    strategy_output_dir = os.path.join(output_dir, output_folder_name)    
    if not os.path.exists(strategy_output_dir):
        os.makedirs(strategy_output_dir)
        
    output_filename = os.path.join(strategy_output_dir, f"{pair}.svg")

    glyph_set1 = font1.getGlyphSet()
    glyph_set2 = font2.getGlyphSet()
    glyph_name1 = font1.getBestCmap().get(ord(char1))
    glyph_name2 = font2.getBestCmap().get(ord(char2))

    if not glyph_name1 or not glyph_name2: return

    pen1_raw = SkiaPathPen(glyph_set1)
    glyph_set1[glyph_name1].draw(pen1_raw)
    path1_raw = pen1_raw.path

    pen2_raw = SkiaPathPen(glyph_set2)
    glyph_set2[glyph_name2].draw(pen2_raw)
    path2_raw = pen2_raw.path

    if uniform_glyphs:
        print("  -> Applying uniform scaling to source glyphs.")
        TARGET_HEIGHT = 1000.0  # Use float for precision, UPM is a good standard

        # Scale path 1 to target height
        bounds1 = path1_raw.bounds
        if bounds1:
            height1 = bounds1[3] - bounds1[1]
            if height1 > 0:
                scale_factor1 = TARGET_HEIGHT / height1
                transform1 = Transform().scale(scale_factor1)
                scaled_pen1 = SkiaPathPen()
                path1_raw.draw(TransformPen(scaled_pen1, transform1))
                path1_raw = scaled_pen1.path

        # Scale path 2 to target height
        bounds2 = path2_raw.bounds
        if bounds2:
            height2 = bounds2[3] - bounds2[1]
            if height2 > 0:
                scale_factor2 = TARGET_HEIGHT / height2
                transform2 = Transform().scale(scale_factor2)
                scaled_pen2 = SkiaPathPen()
                path2_raw.draw(TransformPen(scaled_pen2, transform2))
                path2_raw = scaled_pen2.path
    
    pen2_rotated = SkiaPathPen(glyph_set2)
    if args.noambi:
        path2_rotated = path2_raw
    else:
        path2_raw.draw(TransformPen(pen2_rotated, Transform().rotate(math.pi)))
        path2_rotated = pen2_rotated.path

    if not path1_raw.bounds or not path2_rotated.bounds: return

    merged_skia_path = strategy_func(path1_raw, path2_rotated, pair, alignment_func=alignment_func)
    save_path_as_svg(merged_skia_path, output_filename, glyph_set1)

    if not merged_skia_path:
        print(f"  -> Warning: Strategy '{strategy_name}' failed for '{pair}'. Skipping.")
        return

    bounds = merged_skia_path.bounds
    if not bounds or (bounds[0] == 0 and bounds[1] == 0 and bounds[2] == 0 and bounds[3] == 0): return

    padding = 50
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    
    svg_pen = SVGPathPen(glyph_set1)
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

def create_ambigram_from_string(word1, strategy_name, output_filename, word2=None, target_width=1200, alignment_func=align_using_centroid, uniform_glyphs=False, winning_glyphs=None):
    """
    Creates a single composite ambigram image, scaled to a target width,
    with an option for uniform glyph rendering.
    """
    print(f"\n--- Composing ambigram for '{word1}' / '{word2 if word2 else word1}' ---")
    
    required_pairs = [f"{c1}{c2}" for c1, c2 in zip(word1, word2)]
    glyph_images = []

    render_params = {}
    if uniform_glyphs:
        print("  -> Using uniform glyph height rendering.")
        GLYPH_RENDER_HEIGHT = 256
        render_params['output_height'] = GLYPH_RENDER_HEIGHT
    else:
        print("  -> Using variable (expressive) glyph height rendering.")

    for pair in required_pairs:
        if winning_glyphs:
            winning_alignment = winning_glyphs.get(pair)
            if not winning_alignment:
                print(f"  -> Fatal Error: Could not find a winning glyph for the pair '{pair}'. Aborting composition.")
                return
            # NOTE: Assumes the base strategy is 'outline' when using --select-best
            glyph_dir = os.path.join(".", f"generated_glyphs_outline_{winning_alignment}")
        else:
            glyph_dir = os.path.join(".", f"generated_glyphs_{strategy_name}_{alignment_func.__name__.replace('align_using_','')}")
        
        filename = f"{pair}.svg"
        filepath = os.path.join(glyph_dir, filename)

        if not os.path.exists(filepath):
            print(f"  -> Warning: Required glyph file not found, skipping: {filepath}")
            continue
        try:
            png_data = svg2png(url=filepath, **render_params)
            glyph_image = Image.open(io.BytesIO(png_data))
            glyph_images.append(glyph_image)
            print(f"  -> Loaded and rendered {filename} from {os.path.basename(glyph_dir)}")
        except Exception as e:
            print(f"  -> ERROR: Could not process {filename}. Details: {e}")

    if not glyph_images:
        print("Could not render any glyphs. Aborting composition.")
        return

    total_width = sum(img.width for img in glyph_images)
    max_height = max(img.height for img in glyph_images)
    composite_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 255))
    current_x = 0
    for img in glyph_images:
        # For variable-height glyphs, we align them to the baseline (bottom)
        y_pos = max_height - img.height
        composite_image.paste(img, (current_x, y_pos), img)
        current_x += img.width

    current_width, current_height = composite_image.size
    aspect_ratio = float(current_height) / float(current_width)
    target_height = int(aspect_ratio * target_width)

    print(f"\nResizing final image to {target_width} x {target_height}...")
    final_image = composite_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    final_image.save(output_filename)
    print(f"\nAmbigram saved successfully to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a composite ambigram from one or two words using a specified font.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "word1", 
        type=str, 
        help="The first word to create the ambigram from (reads forwards)."
    )
    parser.add_argument(
        "word2", 
        type=str, 
        nargs='?', 
        default=None, 
        help="(Optional) The second word (reads when rotated 180 degrees).\nIf omitted, a palindromic ambigram of word1 will be created."
    )
    parser.add_argument(
        "-f", "--font", 
        type=str, 
        default="C:\\Windows\\Fonts\\arial.ttf",
        help="Path to the TTF font file to use.\nDefaults to Arial on Windows."
    )
    parser.add_argument(
        "-f2", "--font2", 
        type=str, 
        default=None,
        help="(Optional) Path to a second TTF font file for the second word/character."
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default="outline",
        choices=['outline', 'centerline_trace', 'half_letters'],
        help="The generation strategy to use. 'outline' is a style that can be combined with different alignments. Others are standalone."
    )
    parser.add_argument(
        "-a", "--alignment",
        type=str,
        default="centroid",
        choices=['centroid', 'principal_axis', 'iterative_registration', 'c', 'i', 'p'],
        help="The alignment method to use for strategies that support it (e.g., 'outline')."
    )
    parser.add_argument(
        "--select-best",
        action='store_true',
        help="Automatically test 'outline' with centroid and iterative alignments and select the one with the best legibility score."
    )
    parser.add_argument("-w", "--width", type=int, default=1200, help="The final width of the output PNG image in pixels. Defaults to 1200.")
    parser.add_argument("-u", "--uniform-glyphs", action='store_true', help="If included, renders all glyphs at a uniform height before composition.")
    parser.add_argument("-n", "--noambi", action='store_true', help="Only run the font strategies - do not ambigrammatize. Equivalent to passing word1 in reverse, negating the ambigram effect.")
     
    args = parser.parse_args()

    INPUT_WORD = args.word1
    INPUT_WORD2 = args.word2
    FONT1_FILE_PATH = args.font
    FONT2_FILE_PATH = args.font2
    print(f'DEBUG: FONT1_FILE_PATH {FONT1_FILE_PATH}')
    print(f'DEBUG: FONT2_FILE_PATH {FONT2_FILE_PATH}')
    TARGET_WIDTH = args.width
    UNIFORM_GLYPHS = args.uniform_glyphs

    strategy_map = {
        'centroid': align_using_centroid,
        'principal_axis': align_using_principal_axis,
        'outline': generate_using_outline,
        'centerline_trace': generate_using_centerline_trace,
        'half_letters': generate_using_half_letters,
        'iterative_registration': align_using_iterative_registration
    }
    STRATEGY_TO_USE = strategy_map[args.strategy]

    alignment_map = {
        'c': align_using_centroid,
        'p': align_using_principal_axis,
        'i': align_using_iterative_registration,
    }
    ALIGNMENT_TO_USE = alignment_map[args.alignment[0]]

    if INPUT_WORD2 and len(INPUT_WORD) != len(INPUT_WORD2):
        print(f"ERROR: Input words '{INPUT_WORD}' and '{INPUT_WORD2}' must be the same length.")
        exit()

    if not os.path.exists(FONT1_FILE_PATH):
        print(f"ERROR: Font file not found at '{FONT1_FILE_PATH}'")
        exit()
    
    strategy_name = STRATEGY_TO_USE.__name__.replace('generate_using_', '')
    print(f"--- Using strategy: {strategy_name} ---")

    INPUT_WORD2 = INPUT_WORD2[::-1] if INPUT_WORD2 else INPUT_WORD[::-1]
    if args.noambi:
        INPUT_WORD2 = INPUT_WORD2[::-1]

    pairs_to_generate = list(set([c1 + c2 for c1, c2 in zip(INPUT_WORD, INPUT_WORD2)]))
    
    print(f"Required pairs to generate: {pairs_to_generate}")

    try:
        font1 = TTFont(FONT1_FILE_PATH)
        print(f"DEBUG: {font1['head']}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load font1. Aborting. Details: {e}")
        exit()
    try:
        font2 = TTFont(FONT2_FILE_PATH) if FONT2_FILE_PATH and os.path.exists(FONT2_FILE_PATH) else font1
        print(f"DEBUG: {font2['head']}")
    except Exception as e:
        print(f"WARNING: Could not load font2. Falling back to font1. Details: {e}")

    winning_glyphs = {}

    for pair in pairs_to_generate:
        print(f"\n- Generating glyph for pair: '{pair}'")

        if args.select_best:
            print(f"  -> --select-best enabled: Finding optimal alignment for '{pair}'")

            glyph_set1 = font1.getGlyphSet()
            glyph_name1 = font1.getBestCmap().get(ord(pair[0]))
            pen1_raw = SkiaPathPen(glyph_set1)
            glyph_set1[glyph_name1].draw(pen1_raw)
            path1_raw = pen1_raw.path
            
            glyph_set2 = font2.getGlyphSet()
            glyph_name2 = font2.getBestCmap().get(ord(pair[1]))
            pen2_raw = SkiaPathPen(glyph_set2)
            glyph_set2[glyph_name2].draw(pen2_raw)
            path2_raw = pen2_raw.path

            pen2_rotated = SkiaPathPen(glyph_set2)
            path2_raw.draw(TransformPen(pen2_rotated, Transform().rotate(math.pi)))
            path2_rotated = pen2_rotated.path
            
            print("  -> Testing candidate: outline + centroid")
            path_centroid = generate_using_outline(path1_raw, path2_rotated, pair, alignment_func=align_using_centroid)
            score_centroid = calculate_legibility_score(path_centroid, path1_raw, path2_raw)
            print(f"  -> Score: {score_centroid:.4f}")

            print("  -> Testing candidate: outline + iterative_registration")
            path_iterative = generate_using_outline(path1_raw, path2_rotated, pair, alignment_func=align_using_iterative_registration)
            score_iterative = calculate_legibility_score(path_iterative, path1_raw, path2_raw)
            print(f"  -> Score: {score_iterative:.4f}")

            if score_iterative < score_centroid:
                print("  -> Winner: Iterative Registration")
                best_path = path_iterative
                winning_align_name = "iterative_registration"
            else:
                print("  -> Winner: Centroid")
                best_path = path_centroid
                winning_align_name = "centroid"

            winning_glyphs[pair] = winning_align_name

            
            strategy_output_dir = os.path.join(".", f"generated_glyphs_outline_{winning_align_name}")
            if not os.path.exists(strategy_output_dir):
                os.makedirs(strategy_output_dir)
            output_filename = os.path.join(strategy_output_dir, f"{pair}.svg")
            save_path_as_svg(best_path, output_filename, glyph_set1)

        else:
            generate_ambigram_svg(font1, font2, pair, ".", STRATEGY_TO_USE, alignment_func=ALIGNMENT_TO_USE, uniform_glyphs=UNIFORM_GLYPHS)

    # double reverse string if "noambi")
    output_filename = f"{INPUT_WORD}{'-' + INPUT_WORD2 if args.noambi else INPUT_WORD2[::-1]}_{os.path.basename(FONT1_FILE_PATH)}{'_uni' if UNIFORM_GLYPHS else ''}{'-' + os.path.basename(FONT2_FILE_PATH) if FONT2_FILE_PATH and font1 != font2 else ''}{'-' + args.alignment}_{'no' if args.noambi else ''}ambigram.png"
    create_ambigram_from_string(INPUT_WORD, strategy_name, output_filename, word2=INPUT_WORD2, target_width=TARGET_WIDTH, uniform_glyphs=UNIFORM_GLYPHS, winning_glyphs=winning_glyphs, alignment_func=ALIGNMENT_TO_USE)