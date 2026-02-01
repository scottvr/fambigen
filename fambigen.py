# fambigen.py 
#
# This script generates ambigram glyphs from character pairs (taken from a specified TTF i
# or WOFF) using various strategies and can compose them into a single image based on 
# input words.


# TODO: allow half-letters and centerline_trace in font compositor mode.
# will need to accept alignment_func remove references to undefined globals (glyph_set), 
# and fix the get_vector_skeleton call signature.

import svgwrite
import math
import os
import io
import re
import numpy as np
import traceback
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
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from fontTools.ttLib import newTable


#import faulthandler

import os
from fontTools.ttLib import TTFont

def resolve_font_path_matplotlib(font_arg: str) -> str:
    if not font_arg:
        raise FileNotFoundError("Empty font argument")

    # 1) explicit path
    if os.path.isfile(font_arg):
        return os.path.abspath(font_arg)

    target = font_arg.strip().lower()
    target_base, target_ext = os.path.splitext(target)

    try:
        from matplotlib import font_manager
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for font name resolution. "
            "Install it or pass an explicit font path."
        ) from e

    # 2) enumerate system fonts (dedupe paths)
    font_paths = set()
    for ext in ("ttf", "otf", "ttc"):
        for p in font_manager.findSystemFonts(fontpaths=None, fontext=ext):
            font_paths.add(p)

    # Fast pass: filename match
    filename_hits = set()
    for p in font_paths:
        fn = os.path.basename(p).lower()
        base, ext = os.path.splitext(fn)

        if fn == target:
            filename_hits.add(p)
        elif target_ext == "" and base == target:
            filename_hits.add(p)
        elif target_ext != "" and base == target_base and ext == target_ext:
            filename_hits.add(p)

    if filename_hits:
        hits = sorted(filename_hits)
    else:
        # Slow pass: family match (dedupe)
        family_hits = set()
        for p in font_paths:
            try:
                tt = TTFont(p, lazy=True)
                if "name" not in tt:
                    continue
                for rec in tt["name"].names:
                    if rec.nameID == 1:  # Family
                        fam = rec.toUnicode().strip().lower()
                        if fam == target:
                            family_hits.add(p)
                            break
            except Exception:
                continue
        hits = sorted(family_hits)

    if not hits:
        raise FileNotFoundError(
            f"Could not resolve font '{font_arg}'. "
            "Pass an explicit path or choose an installed font."
        )

    if len(hits) == 1:
        return hits[0]

    # Prefer Regular if unique
    regularish = [p for p in hits if "regular" in os.path.basename(p).lower()]
    if len(regularish) == 1:
        return regularish[0]

    preview = "\n".join(hits[:30])
    more = "" if len(hits) <= 30 else f"\n... ({len(hits) - 30} more)"
    raise FileNotFoundError(
        f"Font '{font_arg}' is ambiguous. Matches:\n{preview}{more}"
    )


def resolve_font_path(font_arg: str) -> str:
    if not font_arg:
        raise FileNotFoundError("Empty font argument")

    if os.path.isfile(font_arg):
        return os.path.abspath(font_arg)

    target = font_arg.strip().lower()
    target_base, target_ext = os.path.splitext(target)

    font_paths = set()
    for ext in ("ttf", "otf", "ttc"):
        for p in font_manager.findSystemFonts(fontpaths=None, fontext=ext):
            font_paths.add(p)

    # Fast pass: filename match
    filename_hits = set()
    for p in font_paths:
        fn = os.path.basename(p).lower()
        base, ext = os.path.splitext(fn)

        if fn == target:
            filename_hits.add(p)
        elif target_ext == "" and base == target:
            filename_hits.add(p)
        elif target_ext != "" and base == target_base and ext == target_ext:
            filename_hits.add(p)

    if filename_hits:
        hits = sorted(filename_hits)
    else:
        # Slow pass: family match (dedupe)
        family_hits = set()
        for p in font_paths:
            try:
                tt = TTFont(p, lazy=True)
                if "name" not in tt:
                    continue
                for rec in tt["name"].names:
                    if rec.nameID == 1:  # Family
                        fam = rec.toUnicode().strip().lower()
                        if fam == target:
                            family_hits.add(p)
                            break
            except Exception:
                continue
        hits = sorted(family_hits)

    if not hits:
        raise FileNotFoundError(
            f"Could not resolve font '{font_arg}'. "
            "Pass an explicit path or choose an installed font."
        )

    if len(hits) == 1:
        return hits[0]

    # Prefer Regular if unique
    regularish = [p for p in hits if "regular" in os.path.basename(p).lower()]
    if len(regularish) == 1:
        return regularish[0]

    preview = "\n".join(hits[:30])
    more = "" if len(hits) <= 30 else f"\n... ({len(hits) - 30} more)"
    raise FileNotFoundError(
        f"Font '{font_arg}' is ambiguous. Matches:\n{preview}{more}"
    )

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

def skia_path_to_contours(skia_path, segment_len=20):
    """
    Convert SkiaPath iteration into polygon contours.
    Returns: list of contours; each contour is a list of (x,y) tuples.
    NOTE: This is a first-pass polygonization (line segments only).
    """
    if not skia_path:
        return []

    rec = RecordingPen()
    flat = FlattenPen(rec, approximateSegmentLength=segment_len)
    skia_path.draw(flat)

    contours = []
    current = []

    for verb, pts in rec.value:
        if verb == "moveTo":
            if current:
                contours.append(current)
                current = []
            current.append(tuple(pts[0]))
        elif verb in ("lineTo", "qCurveTo", "curveTo"):
            # FlattenPen should have converted curves into line segments,
            # but keep this robust: take last point
            current.append(tuple(pts[-1]))
        elif verb == "closePath":
            if current:
                contours.append(current)
                current = []

    if current:
        contours.append(current)

    # Filter junk
    contours = [c for c in contours if len(c) >= 3]
    return contours


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

from fontTools.pens.transformPen import TransformPen

def skia_path_to_ttglyph(skia_path, glyph_set=None):
    pen = TTGlyphPen(glyph_set)
    skia_path.draw(pen)
    return pen.glyph()

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


def replace_post_with_format3(tt):
    post = newTable("post")
    post.formatType = 3.0
    post.italicAngle = 0
    post.underlinePosition = -75
    post.underlineThickness = 50
    post.isFixedPitch = 0
    post.minMemType42 = 0
    post.maxMemType42 = 0
    post.minMemType1 = 0
    post.maxMemType1 = 0
    tt["post"] = post

def build_cmap(cp_to_glyph):
    cmap = newTable("cmap")
    cmap.tableVersion = 0
    cmap.tables = []

    sub = CmapSubtable.newSubtable(4)
    sub.platformID = 3
    sub.platEncID = 1
    sub.language = 0
    sub.cmap = dict(cp_to_glyph)
    cmap.tables.append(sub)

    sub2 = CmapSubtable.newSubtable(4)
    sub2.platformID = 0
    sub2.platEncID = 3
    sub2.language = 0
    sub2.cmap = dict(cp_to_glyph)
    cmap.tables.append(sub2)

    return cmap

def emit_composite_font(
    out_path,
    font1,
    font2,
    charset_codepoints,
    strategy_func,
    alignment_func,
    uniform_glyphs=False,
    width_mode="max",
    family_name=None,
    style_name="Regular",
    version="1.000",
    vendor="FMBG",
    build_id="",
    ):


    out = TTFont(FONT1_FILE_PATH) 

    for tag in ["kern", "GPOS", "GSUB", "DSIG"]:
        if tag in out:
            del out[tag]

    gs1 = font1.getGlyphSet()
    gs2 = font2.getGlyphSet()

    glyph_order = [".notdef"]
    cp_to_glyph = {}
    glyph_contours = {}     
    advance_widths = {}     
    new_glyf = {}
    new_hmtx = {}

    pen = TTGlyphPen(None)
    new_glyf[".notdef"] = pen.glyph()  
    new_hmtx[".notdef"] = (500, 0)    

    # Pull widths from font1 hmtx if available
    hmtx1 = font1["hmtx"].metrics if "hmtx" in font1 else {}
    
    def _apply_uniform_scale(path):
        if not path or not path.bounds:
            return path
        TARGET_HEIGHT = 1000.0
        b = path.bounds
        h = b[3] - b[1]
        if h <= 0:
            return path
        scale_factor = TARGET_HEIGHT / float(h)
        t = Transform().scale(scale_factor)
        p = SkiaPathPen()
        path.draw(TransformPen(p, t))
        return p.path

    def _advance_width_for(cp, gname1, gname2, merged_path):
        # Default LSB = 0 for now (crude but acceptable v1)
        adv1 = None
        adv2 = None
        try:
            adv1 = font1["hmtx"][gname1][0]
        except Exception:
            pass
        try:
            adv2 = font2["hmtx"][gname2][0]
        except Exception:
            pass

        if width_mode == "font1":
            adv = adv1 if adv1 is not None else (adv2 if adv2 is not None else 600)
            return (int(adv), 0)

        if width_mode == "max":
            candidates = [a for a in [adv1, adv2] if a is not None]
            adv = max(candidates) if candidates else 600
            return (int(adv), 0)

        if width_mode == "auto":
            # Compute from merged bounds. Add padding.
            if merged_path and merged_path.bounds:
                b = merged_path.bounds
                w = b[2] - b[0]
                pad = max(20.0, w * 0.10)
                adv = int(max(200.0, w + pad))
                return (adv, 0)
            return (600, 0)

        # fallback
        return (600, 0)

    for cp in charset_codepoints:
        ch = chr(cp)

        g1 = font1.getBestCmap().get(cp)
        g2 = font2.getBestCmap().get(cp)

        if not g1 and not g2:
            continue

        out_g = g1 or g2 or f"uni{cp:04X}"

        # Avoid duplicates in glyphOrder
        if out_g not in glyph_order:
            glyph_order.append(out_g)

        cp_to_glyph[cp] = out_g

        # Get raw paths from each font for this codepoint (fallback to whichever exists)
        path1_raw = None
        path2_raw = None

        if g1:
            pen1 = SkiaPathPen(gs1); gs1[g1].draw(pen1); path1_raw = pen1.path
        if g2:
            pen2 = SkiaPathPen(gs2); gs2[g2].draw(pen2); path2_raw = pen2.path

        # If one missing, just use the other (no composite)
        if path1_raw and not path2_raw:
            merged = path1_raw
        elif path2_raw and not path1_raw:
            merged = path2_raw

        if uniform_glyphs:
            path1_raw = _apply_uniform_scale(path1_raw)
            path2_raw = _apply_uniform_scale(path2_raw)

        merged = strategy_func(path1_raw, path2_raw, pair=f"{ch}{ch}", alignment_func=alignment_func)

        if merged is None:
            pen = TTGlyphPen(None)
            ttglyph = pen.glyph()
        else:
            ttglyph = skia_path_to_ttglyph(merged)  
        
        new_glyf[out_g] = ttglyph
        new_hmtx[out_g] = _advance_width_for(cp, g1, g2, merged)    # Option B: Process the valid path


    out.setGlyphOrder(glyph_order)
    
    out["glyf"].glyphs = new_glyf 
    hmtx_table = newTable("hmtx")
    hmtx_table.metrics = new_hmtx   
    out["hmtx"] = hmtx_table

    for g in glyph_order:
        out["glyf"][g].recalcBounds(out["glyf"])
    
    out["maxp"].numGlyphs = len(glyph_order)
    if hasattr(out["maxp"], "recalculate"):
        out["maxp"].recalculate(glyfTable=out["glyf"])
    if "hhea" in out:
        out["hhea"].numberOfHMetrics = len(glyph_order)
    out["cmap"] = build_cmap(cp_to_glyph)

    replace_post_with_format3(out)

    set_font_metadata(
        out,
        family=(family_name or default_family_name_from_outpath(out_path)),
        style=style_name,
        version=version,
        vendor=vendor,
        build_id=build_id,
    )
    
    def set_output_format_from_extension(ttfont_obj, out_path):
        ext = os.path.splitext(out_path)[1].lower()
    
        if ext == ".ttf":
            ttfont_obj.flavor = None
            # TrueType outlines (glyf) already.
            return
    
        if ext == ".woff":
            ttfont_obj.flavor = "woff"
            return
        
        # will require brotli
        #if ext == ".woff2":
        #    ttfont_obj.flavor = "woff2"
        #    return
    
        if ext == ".otf":
            raise ValueError(
                "OTF output not yet supported.\n"
                "This generator builds TrueType (glyf) outlines; real .otf usually requires CFF conversion. "
                "Use .ttf/.woff"
            )
    
        raise ValueError(f"Unknown output extension '{ext}'. Use .ttf/.woff/.woff2.")

    # Save
    set_output_format_from_extension(out, out_path)
    out.save(out_path)


def OLD_emit_composite_font(
    out_path,
    font1,
    font2,
    charset_codepoints,
    strategy_func,
    alignment_func,
    uniform_glyphs=False,
    width_mode="max",
    family_name=None,
    style_name="Regular",
    version="1.000",
    vendor="FMBG",
    build_id="",
    ):  
    """
    Emit a subset TrueType/WOFF font consisting of composite glyphs:
      out_glyph(c) = merge( glyph1(c), glyph2(c) )

    No ambigram rotation is performed. This is compositing only.
    """

    # Start from font1 as a template. This preserves a lot of tables that
    # make the font installable, and we replace cmap/glyf/hmtx appropriately.
    out = TTFont()  # build a fresh font for clarity

    # Copy required tables from font1 when present
    for tag in ["head", "hhea", "maxp", "OS/2", "post", "name", "loca"]:
        if tag in font1:
            out[tag] = font1[tag]

    # glyf/hmtx/cmap will be rebuilt
    out["glyf"] = font1["glyf"] if "glyf" in font1 else None
    out["hmtx"] = font1["hmtx"] if "hmtx" in font1 else None

    glyph_set1 = font1.getGlyphSet()
    glyph_set2 = font2.getGlyphSet()

    cmap1 = font1.getBestCmap()
    cmap2 = font2.getBestCmap()

    # Always include .notdef as first glyph
    glyph_order = [".notdef"]
    glyf_table = out["glyf"]
    hmtx_table = out["hmtx"]

    # Build new glyph objects and metrics
    new_glyf = {}
    new_hmtx = {}

    # Create an empty .notdef (or reuse if it exists in font1)
    if ".notdef" in getattr(font1, "getGlyphOrder", lambda: [])():
        try:
            new_glyf[".notdef"] = font1["glyf"][".notdef"]
            new_hmtx[".notdef"] = font1["hmtx"][".notdef"]
        except Exception:
            pen = TTGlyphPen(None)
            new_glyf[".notdef"] = pen.glyph()
            new_hmtx[".notdef"] = (500, 0)
    else:
        pen = TTGlyphPen(None)
        new_glyf[".notdef"] = pen.glyph()
        new_hmtx[".notdef"] = (500, 0)

    # Helper: uniform scaling copied from your generate_ambigram_svg() logic
    def _apply_uniform_scale(path):
        if not path or not path.bounds:
            return path
        TARGET_HEIGHT = 1000.0
        b = path.bounds
        h = b[3] - b[1]
        if h <= 0:
            return path
        scale_factor = TARGET_HEIGHT / float(h)
        t = Transform().scale(scale_factor)
        p = SkiaPathPen()
        path.draw(TransformPen(p, t))
        return p.path

    def _advance_width_for(cp, gname1, gname2, merged_path):
        # Default LSB = 0 for now (crude but acceptable v1)
        adv1 = None
        adv2 = None
        try:
            adv1 = font1["hmtx"][gname1][0]
        except Exception:
            pass
        try:
            adv2 = font2["hmtx"][gname2][0]
        except Exception:
            pass

        if width_mode == "font1":
            adv = adv1 if adv1 is not None else (adv2 if adv2 is not None else 600)
            return (int(adv), 0)

        if width_mode == "max":
            candidates = [a for a in [adv1, adv2] if a is not None]
            adv = max(candidates) if candidates else 600
            return (int(adv), 0)

        if width_mode == "auto":
            # Compute from merged bounds. Add padding.
            if merged_path and merged_path.bounds:
                b = merged_path.bounds
                w = b[2] - b[0]
                pad = max(20.0, w * 0.10)
                adv = int(max(200.0, w + pad))
                return (adv, 0)
            return (600, 0)

        # fallback
        return (600, 0)

    # Generate composite glyphs
    for cp in charset_codepoints:
        gname1 = cmap1.get(cp)
        if not gname1:
            continue  # skip chars font1 doesn't support
        gname2 = cmap2.get(cp) or gname1

        # Extract outlines into SkiaPath
        try:
            pen1 = SkiaPathPen(glyph_set1)
            glyph_set1[gname1].draw(pen1)
            path1_raw = pen1.path

            pen2 = SkiaPathPen(glyph_set2)
            glyph_set2[gname2].draw(pen2)
            path2_raw = pen2.path
        except Exception:
            continue

        if not path1_raw or not path1_raw.bounds:
            continue

        if uniform_glyphs:
            path1_raw = _apply_uniform_scale(path1_raw)
            path2_raw = _apply_uniform_scale(path2_raw)

        # NO ROTATION. compositing only.
        pair = chr(cp) + chr(cp)

        # For emission, restrict to outline strategy unless you want to harden others.
        try:
            merged = _call_strategy(strategy_func, path1_raw, path2_raw, pair, alignment_func)
        except TypeError:
            # Strategy function signature mismatch; skip.
            continue
        except Exception:
            continue

        if not merged or not merged.bounds:
            continue

        # Convert merged SkiaPath to a TrueType glyph (flattened)
        ttglyph = skia_path_to_ttglyph(merged, glyph_name_for_debug=gname1)

        # Use original glyph name from font1 to keep cmap mapping simple
        out_gname = gname1
        new_glyf[out_gname] = ttglyph
        new_hmtx[out_gname] = _advance_width_for(cp, gname1, gname2, merged)

        if out_gname not in glyph_order:
            glyph_order.append(out_gname)

    out.setGlyphOrder(glyph_order)

    # Ensure glyf/hmtx exist
    if "glyf" not in out:
        out["glyf"] = font1["glyf"]
    if "hmtx" not in out:
        out["hmtx"] = font1["hmtx"]

    # Write glyf data
    out["glyf"].glyphs = {}
    for g in glyph_order:
        out["glyf"].glyphs[g] = new_glyf.get(g, TTGlyphPen(None).glyph())

    # Write hmtx metrics
    out["hmtx"].metrics = {}
    for g in glyph_order:
        out["hmtx"].metrics[g] = new_hmtx.get(g, (600, 0))

    # Recalc bounds for each glyph
    try:
        for g in glyph_order:
            out["glyf"][g].recalcBounds(out["glyf"])
    except Exception:
        pass

    # Build a minimal cmap (format 4, BMP only)
    cmap_table = out.get("cmap", None)
    if cmap_table is None:
        out["cmap"] = font1["cmap"]
        cmap_table = out["cmap"]

    sub = CmapSubtable.newSubtable(4)
    sub.platformID = 3
    sub.platEncID = 1
    sub.language = 0
    sub.cmap = {}

    for cp in charset_codepoints:
        if cp > 0xFFFF:
            continue
        gname = cmap1.get(cp)
        if not gname:
            continue
        if gname in new_glyf:
            sub.cmap[cp] = gname

    cmap_table.tables = [sub]

    # Update maxp / hhea counts
    if "maxp" in out:
        out["maxp"].numGlyphs = len(glyph_order)
    if "hhea" in out:
        out["hhea"].numberOfHMetrics = len(glyph_order)

    # Update name table (optional)
    if family_name and "name" in out:
        # Very light-touch: set family name (nameID 1) and full name (nameID 4)
        # across existing records where applicable.
        for rec in out["name"].names:
            try:
                if rec.nameID == 1:
                    rec.string = family_name.encode(rec.getEncoding(), errors="replace")
                if rec.nameID == 4:
                    rec.string = (family_name + " Regular").encode(rec.getEncoding(), errors="replace")
            except Exception:
                pass

    def set_output_format_from_extension(ttfont_obj, out_path):
        ext = os.path.splitext(out_path)[1].lower()
    
        if ext == ".ttf":
            ttfont_obj.flavor = None
            # TrueType outlines (glyf) already.
            return
    
        if ext == ".woff":
            ttfont_obj.flavor = "woff"
            return
        
        # will require brotli
        #if ext == ".woff2":
        #    ttfont_obj.flavor = "woff2"
        #    return
    
        if ext == ".otf":
            raise ValueError(
                "OTF output not yet supported.\n"
                "This generator builds TrueType (glyf) outlines; real .otf usually requires CFF conversion. "
                "Use .ttf/.woff"
            )
    
        raise ValueError(f"Unknown output extension '{ext}'. Use .ttf/.woff/.woff2.")

    # Save
    set_output_format_from_extension(out, out_path)
    set_font_metadata(
        out,
        family=(family_name or default_family_name_from_outpath(out_path)),
        style=style_name,
        version=version,
        vendor=vendor,
        build_id=build_id,
    )
    out.save(out_path)
    print(f"Saved composite font to: {out_path}")

def _ps_sanitize(s: str) -> str:
    # PostScript name: no spaces; keep ASCII-ish [A-Za-z0-9_-]
    s = s.replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9_-]+", "", s)
    return s or "Font"

def set_font_metadata(
    tt: TTFont,
    family: str,
    style: str,
    version: str,
    vendor: str,
    build_id: str,
):
    """
    Rewrite naming/IDs so the emitted font won't collide with the source.
    build_id is appended when non-empty.
    """

    family_menu = family.strip()
    style_menu = (style or "Regular").strip()

    # Add cache-bust suffix in a controlled way
    if build_id:
        family_menu_busted = f"{family_menu} {build_id}"
    else:
        family_menu_busted = family_menu

    full_name = f"{family_menu_busted} {style_menu}".strip()

    ps_family = _ps_sanitize(family_menu_busted)
    ps_style  = _ps_sanitize(style_menu)
    ps_name = f"{ps_family}-{ps_style}" if ps_style else ps_family

    version_str = f"Version {version}"
    vendor4 = (vendor[:4].ljust(4, " "))

    # Unique ID should be unique across installs; include vendor + build_id + version.
    # Keep it deterministic unless --cache-bust timestamp.
    unique_id = f"{vendor4};{family_menu_busted};{style_menu};{version}"

    if "name" not in tt:
        raise RuntimeError("Output font has no 'name' table to edit.")

    name_table = tt["name"]

    replacements = {
        1: family_menu_busted,   # Family
        2: style_menu,           # Subfamily
        3: unique_id,            # Unique identifier
        4: full_name,            # Full name
        5: version_str,          # Version string
        6: ps_name,              # PostScript name (critical)
        16: family_menu_busted,  # Typographic family (recommended)
        17: style_menu,          # Typographic subfamily
    }

    # Overwrite existing records
    for rec in name_table.names:
        if rec.nameID in replacements:
            text = replacements[rec.nameID]
            try:
                rec.string = text.encode(rec.getEncoding(), errors="replace")
            except Exception:
                rec.string = text.encode("utf-16be", errors="replace")

    # Add missing records for Windows + Mac
    existing = {(rec.nameID, rec.platformID, rec.platEncID, rec.langID) for rec in name_table.names}
    add_targets = [
        (3, 1, 0x0409),  # Windows, Unicode BMP, en-US
        (1, 0, 0),       # Mac, Roman, English
    ]
    for name_id, text in replacements.items():
        for platformID, platEncID, langID in add_targets:
            key = (name_id, platformID, platEncID, langID)
            if key in existing:
                continue
            try:
                name_table.setName(text, name_id, platformID, platEncID, langID)
            except Exception:
                pass

    # Vendor ID
    if "OS/2" in tt:
        try:
            tt["OS/2"].achVendID = vendor4.encode("ascii", errors="replace")
        except Exception:
            pass

    # Revision (helps cache behavior in some stacks)
    if "head" in tt:
        try:
            tt["head"].fontRevision = float(version)
        except Exception:
            pass

#def set_font_metadata(tt, family, style="Regular", version="1.000", vendor="FMBG"):
#    """
#    Rewrite name table + a couple identifiers so the emitted font does not
#    collide with the source font (e.g. Arial).
#    """
#
#    # PostScript name rules: no spaces, typically ASCII
#    ps_family = "".join(ch for ch in family if ch.isalnum() or ch in "-_")
#    ps_style  = "".join(ch for ch in style  if ch.isalnum() or ch in "-_")
#    ps_name = f"{ps_family}-{ps_style}" if ps_style else ps_family
#
#    full_name = f"{family} {style}".strip()
#    version_str = f"Version {version}"
#
#    # Make a unique ID that changes per build (simple deterministic option):
#    # include family/style/version + vendor. You can also add a timestamp if you want.
#    unique_id = f"{vendor};{family};{style};{version}"
#
#    if "name" not in tt:
#        # If you built from scratch and forgot name table, copy from a template or create.
#        raise RuntimeError("Output font has no 'name' table to edit.")
#
#    name_table = tt["name"]
#
#    replacements = {
#        1: family,
#        2: style,
#        3: unique_id,
#        4: full_name,
#        5: version_str,
#        6: ps_name,
#        16: family,
#        17: style,
#    }
#
#    # Overwrite existing records for these IDs across all platforms/encodings
#    for rec in name_table.names:
#        if rec.nameID in replacements:
#            try:
#                rec.string = replacements[rec.nameID].encode(rec.getEncoding(), errors="replace")
#            except Exception:
#                # As a fallback, write UTF-16BE for Windows records
#                rec.string = replacements[rec.nameID].encode("utf-16be", errors="replace")
#
#    # Ensure missing required records exist (some fonts won't have 16/17 for example)
#    existing = {(rec.nameID, rec.platformID, rec.platEncID, rec.langID) for rec in name_table.names}
#
#    # Add at least Windows Unicode BMP (platform 3, enc 1, lang 0x0409) and Mac Roman (platform 1, enc 0, lang 0)
#    add_targets = [
#        (3, 1, 0x0409),  # Windows, Unicode BMP, en-US
#        (1, 0, 0),       # Mac, Roman, English
#    ]
#
#    for name_id, text in replacements.items():
#        for platformID, platEncID, langID in add_targets:
#            key = (name_id, platformID, platEncID, langID)
#            if key in existing:
#                continue
#            try:
#                name_table.setName(text, name_id, platformID, platEncID, langID)
#            except Exception:
#                pass
#
#    # Vendor ID
#    if "OS/2" in tt:
#        vend = (vendor[:4].ljust(4, " ")).encode("ascii", errors="replace")
#        try:
#            tt["OS/2"].achVendID = vend
#        except Exception:
#            pass
#
#    # Revision
#    if "head" in tt:
#        try:
#            tt["head"].fontRevision = float(version)
#        except Exception:
#            pass

def default_family_name_from_outpath(out_path: str) -> str:
    base = os.path.splitext(os.path.basename(out_path))[0]
    base = re.sub(r"[._]+", " ", base).strip()
    base = re.sub(r"\s+", " ", base)
    return base or "FambigenComposite"

def make_postscript_name(family: str, style: str) -> str:
    def clean(s):
        s = re.sub(r"[^A-Za-z0-9_-]+", "", s.replace(" ", ""))
        return s or "Font"
    return f"{clean(family)}-{clean(style)}"
import hashlib
import time

def compute_build_id(recipe_text: str, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "timestamp":
        return time.strftime("%Y%m%d%H%M%S")
    # mode == "hash"
    h = hashlib.sha1(recipe_text.encode("utf-8")).hexdigest()
    return h[:8]  # short, readable

def make_recipe_string(
    font1_path: str,
    font2_path: str,
    strategy_name: str,
    alignment_name: str,
    charset_name: str,
    uniform_glyphs: bool,
    width_mode: str,
    version: str,
) -> str:
    # Make it stable and explicit. Paths are OK; basenames are also fine.
    # If you want cross-machine determinism, use basenames only.
    return "\n".join([
        f"font1={os.path.basename(font1_path)}",
        f"font2={os.path.basename(font2_path)}",
        f"strategy={strategy_name}",
        f"alignment={alignment_name}",
        f"charset={charset_name}",
        f"uniform_glyphs={int(bool(uniform_glyphs))}",
        f"width_mode={width_mode}",
        f"version={version}",
    ])


def _call_strategy(strategy_func, path1, path2, pair, alignment_func):
    # Some strategies take alignment_func, others do not.
    try:
        return strategy_func(path1, path2, pair, alignment_func=alignment_func)
    except TypeError:
        # Fallback for strategies that don't accept alignment_func
        return strategy_func(path1, path2, pair)


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
    
    parser.add_argument(
        "--emit-font",
        type=str,
        default=None,
        help="Emit an installable font (TTF/WOFF) in compositing-only mode. Example: --emit-font out.ttf"
    )
    parser.add_argument(
        "--charset",
        type=str,
        default="ascii",
        choices=["ascii", "latin1", "input"],
        help="Which codepoints to emit when using --emit-font."
    )
    parser.add_argument(
        "--width-mode",
        type=str,
        default="max",
        choices=["font1", "max", "auto"],
        help="How to choose advance widths in emitted font."
    )
    parser.add_argument(
      "--family-name",
      type=str,
      default=None,
      help="Override font family name stored in the emitted font's name table. "
         "Default: derived from --emit-font output filename basename."
    )

    parser.add_argument(
      "--cache-bust",
      type=str,
      default="hash",
      choices=["none", "hash", "timestamp"],
      help="How to avoid OS font cache/name collisions for emitted fonts. "
         "none=stable names, hash=append deterministic build id (default), "
         "timestamp=append time-based id."
    )

    parser.add_argument(
      "--version",
      type=str,
      default="1.000",
      help="Font version string used in name table and head.fontRevision (default: 1.000)."
    )

    parser.add_argument(
      "--vendor",
      type=str,
      default="FMBG",
      help="4-char vendor tag for OS/2.achVendID (default: FMBG)."
    )

    args = parser.parse_args()

    INPUT_WORD = args.word1
    INPUT_WORD2 = args.word2

    try:
      FONT1_FILE_PATH = resolve_font_path_matplotlib(args.font)
    except Exception as e:
      print(f"ERROR: {e}")
      exit(1)

    if args.font2:
      try:
        FONT2_FILE_PATH = resolve_font_path_matplotlib(args.font2)
      except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
    else:
      FONT2_FILE_PATH = None

    
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

    if args.emit_font:
        # Force compositing behavior
        supported_emit_strategies = {"outline"}  # expand later
        if args.strategy not in supported_emit_strategies:
            print("ERROR: --emit-font currently supports only --strategy outline.")
            print("       (Other strategies are not yet wired/robust for font emission.)")
            exit(1)

        # Build charset
        if args.charset == "ascii":
            cps = list(range(0x20, 0x7F))
        elif args.charset == "latin1":
            cps = list(range(0x20, 0x100))
        elif args.charset == "input":
            s = INPUT_WORD
            cps = sorted(set(ord(ch) for ch in s))
        else:
            cps = list(range(0x20, 0x7F))

        if args.emit_font and args.charset != "input":
            print("NOTE: --charset is not 'input'; positional word argument will be ignored.")

        if args.emit_font and args.charset == "input" and INPUT_WORD == "":
            print("ERROR: --charset input requires a non-empty word to define the glyph subset.")
            exit(1)

        family = args.family_name or default_family_name_from_outpath(args.emit_font)
        recipe = make_recipe_string(
            font1_path=FONT1_FILE_PATH,
            font2_path=(FONT2_FILE_PATH or FONT1_FILE_PATH),
            strategy_name=args.strategy,
            alignment_name=args.alignment,
            charset_name=args.charset,
            uniform_glyphs=UNIFORM_GLYPHS,
            width_mode=args.width_mode,
            version=args.version,
        )
        build_id = compute_build_id(recipe, args.cache_bust)

        emit_composite_font(
            out_path=args.emit_font,
            font1=font1,
            font2=font2,
            charset_codepoints=cps,
            strategy_func=STRATEGY_TO_USE,
            alignment_func=ALIGNMENT_TO_USE,
            uniform_glyphs=UNIFORM_GLYPHS,
            width_mode=args.width_mode,
            family_name=family,
            style_name="Regular",
            version=args.version,
            vendor=args.vendor,
            build_id=build_id,
        )
        exit(0)

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