"""
Advanced Cobb Angle Calculator for SpineCheck-AI
Improved algorithm to automatically detect and exclude outlier vertebrae
that are distant from the main spine curve.
"""

import numpy as np
import cv2
import math
from scipy.ndimage import label, find_objects, center_of_mass
from scipy import stats

# ---- Visualization style (top-level constants) ----
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.85
TEXT_THICK = 2
TEXT_COLOR = (0, 200, 255)    # amber
TEXT_SHADOW = (0, 0, 0)

CURVE_COLOR = (255, 255, 0)   # spine curve
CURVE_THICK = 3               # ↑ was 2

ANGLE_COLOR = (0, 0, 255)     # Cobb arms
ANGLE_THICK = 4               # ↑ was 2

INCLUDED_DOT = (0, 255, 0)    # included vertebrae
END_DOT = (0, 165, 255)       # end vertebrae
OUTLIER_DOT = (0, 0, 255)     # outliers
DOT_RADIUS = 5


def _draw_text(img, text, org, color=TEXT_COLOR):
    """Readable text with soft shadow."""
    cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                FONT, TEXT_SCALE, TEXT_SHADOW, TEXT_THICK + 2, cv2.LINE_AA)
    cv2.putText(img, text, org,
                FONT, TEXT_SCALE, color, TEXT_THICK, cv2.LINE_AA)



class VertebraInfo:
    """Class to store vertebra information"""

    def __init__(self, id, mask, center, bbox):
        self.id = id
        self.mask = mask
        self.center = center  # (y, x) format
        self.bbox = bbox  # (y_min, x_min, y_max, x_max)
        self.contour = None
        self.area = 0
        self.superior_endplate_slope = None  # M1
        self.inferior_endplate_slope = None  # M2
        self.superior_endplate_points = []  # Points for superior endplate
        self.inferior_endplate_points = []  # Points for inferior endplate
        self.is_end_vertebra = False  # Flag to mark top or bottom vertebrae
        self.is_outlier = False  # Flag to mark vertebrae that are outliers from the main curve
        self.distance_from_curve = 0.0  # Distance from the vertebra to the spine curve


def extract_green_mask(image):
    """Extract green highlighted areas from the input image"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    # Threshold the HSV image to get only green colors
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to clean mask
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return green_mask


def segment_vertebrae(mask, min_area=100):
    """Segment spine mask into individual vertebrae."""
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Check if mask is empty
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask), 0

    # Use connected components to label vertebrae
    labeled_mask, num_vertebrae = label(binary_mask)

    # Filter small components
    for i in range(1, num_vertebrae + 1):
        component = (labeled_mask == i)
        if np.sum(component) < min_area:
            labeled_mask[component] = 0

    # Relabel to ensure consecutive labels
    if num_vertebrae > 0:
        temp_mask = np.zeros_like(labeled_mask)
        new_label = 1
        for i in range(1, num_vertebrae + 1):
            component = (labeled_mask == i)
            if np.sum(component) > 0:  # Skip labels that were zeroed
                temp_mask[component] = new_label
                new_label += 1

        labeled_mask = temp_mask
        num_vertebrae = new_label - 1

    return labeled_mask, num_vertebrae


def algorithm1_minimum_enclosing_rectangle(vertebra_info):
    """
    Algorithm 1: Calculate the minimum enclosing rectangle for a vertebra
    and determine superior and inferior endplate points.
    """
    # Prepare vertebra mask
    mask = vertebra_info.mask.astype(np.uint8)

    # Find vertebra contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return vertebra_info

    contour = max(contours, key=cv2.contourArea)
    vertebra_info.contour = contour
    vertebra_info.area = cv2.contourArea(contour)

    # Calculate minimum area rectangle
    min_rect = cv2.minAreaRect(contour)

    # Get the four corner coordinates of the rectangle
    rect_points = cv2.boxPoints(min_rect)

    # Sort points by y-coordinate (top to bottom)
    rect_points = rect_points[rect_points[:, 1].argsort()]

    # The top two points belong to superior endplate
    # The bottom two points belong to inferior endplate
    superior_points = rect_points[:2]
    inferior_points = rect_points[2:]

    # Sort each pair by x-coordinate for consistent ordering
    superior_points = superior_points[superior_points[:, 0].argsort()]
    inferior_points = inferior_points[inferior_points[:, 0].argsort()]

    # Perform interpolation to obtain more coordinate points
    # For superior endplate
    x1, y1 = superior_points[0]
    x2, y2 = superior_points[1]

    # Create 5 interpolation points along the line
    for i in range(5):
        t = i / 4.0  # Parameter from 0 to 1
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        vertebra_info.superior_endplate_points.append((x, y))

    # For inferior endplate
    x1, y1 = inferior_points[0]
    x2, y2 = inferior_points[1]

    # Create 5 interpolation points along the line
    for i in range(5):
        t = i / 4.0  # Parameter from 0 to 1
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        vertebra_info.inferior_endplate_points.append((x, y))

    # Fit the coordinates using least square method
    # For superior endplate
    superior_xs = [p[0] for p in vertebra_info.superior_endplate_points]
    superior_ys = [p[1] for p in vertebra_info.superior_endplate_points]

    if len(superior_xs) >= 2:
        try:
            slope, _, _, _, _ = stats.linregress(superior_xs, superior_ys)
            vertebra_info.superior_endplate_slope = slope
        except:
            pass

    # For inferior endplate
    inferior_xs = [p[0] for p in vertebra_info.inferior_endplate_points]
    inferior_ys = [p[1] for p in vertebra_info.inferior_endplate_points]

    if len(inferior_xs) >= 2:
        try:
            slope, _, _, _, _ = stats.linregress(inferior_xs, inferior_ys)
            vertebra_info.inferior_endplate_slope = slope
        except:
            pass

    return vertebra_info


def fit_spine_curve(vertebrae):
    """Fit a polynomial curve to the vertebrae centers to represent the spine curve."""
    # Extract centers
    centers = np.array([(v.center[1], v.center[0]) for v in vertebrae])  # (x, y) format

    # Sort centers by y-coordinate (top to bottom)
    centers = centers[centers[:, 1].argsort()]

    # Extract x and y coordinates
    x_coords = centers[:, 0]
    y_coords = centers[:, 1]

    # Fit a 3rd degree polynomial (cubic) to represent the spine curve
    try:
        poly_coeffs = np.polyfit(y_coords, x_coords, 3)
        y_values = np.linspace(min(y_coords), max(y_coords), 100)
        return poly_coeffs, y_values
    except:
        return None, None


def identify_outliers(vertebrae):
    """Identify vertebrae that are outliers from the main spine curve."""
    if len(vertebrae) < 5:  # Need enough vertebrae to establish a curve
        return vertebrae

    # Fit a spine curve
    poly_coeffs, y_values = fit_spine_curve(vertebrae)

    if poly_coeffs is None:
        return vertebrae

    # Create spine curve function
    spine_curve = np.poly1d(poly_coeffs)

    # Calculate distance from each vertebra to the curve
    distances = []
    for vertebra in vertebrae:
        # Get center coordinates
        y, x = vertebra.center  # Original format (y, x)

        # Calculate expected x-coordinate based on the spine curve
        expected_x = spine_curve(y)

        # Calculate distance from vertebra center to curve
        distance = abs(x - expected_x)

        # Store distance
        vertebra.distance_from_curve = distance
        distances.append(distance)

    # Calculate threshold for outlier detection
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Mark vertebrae as outliers if they are more than 2 standard deviations from the mean
    threshold = mean_distance + 2 * std_distance

    for vertebra in vertebrae:
        if vertebra.distance_from_curve > threshold:
            vertebra.is_outlier = True

    return vertebrae, poly_coeffs, y_values


def extract_vertebrae(mask):
    """
    Extract individual vertebrae from segmentation mask and analyze them
    using Algorithm 1. Marks topmost, bottommost, and outlier vertebrae
    to be excluded from angle calculations.
    """
    # Ensure mask is single channel
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Ensure binary mask
    binary_mask = (mask > 0).astype(np.uint8)

    # Segment vertebrae
    labeled_mask, num_vertebrae = segment_vertebrae(binary_mask)

    if num_vertebrae == 0:
        return []

    # Analyze vertebrae
    vertebrae = []

    for i in range(1, num_vertebrae + 1):
        # Create mask for this vertebra
        vertebra_mask = (labeled_mask == i)

        # Skip small components (likely noise)
        if np.sum(vertebra_mask) < 100:
            continue

        # Find bounding box
        regions = find_objects(vertebra_mask)
        if not regions:
            continue

        bbox_slice = regions[0]
        y_min, x_min = bbox_slice[0].start, bbox_slice[1].start
        y_max, x_max = bbox_slice[0].stop, bbox_slice[1].stop

        # Calculate center of mass
        cy, cx = center_of_mass(vertebra_mask)

        # Create vertebra info
        vertebra = VertebraInfo(
            id=i,
            mask=vertebra_mask,
            center=(cy, cx),
            bbox=(y_min, x_min, y_max, x_max)
        )

        # Apply Algorithm 1: Calculate minimum enclosing rectangle and endplates
        vertebra = algorithm1_minimum_enclosing_rectangle(vertebra)

        # Add to list
        vertebrae.append(vertebra)

    # Sort vertebrae by y-coordinate (top to bottom)
    vertebrae.sort(key=lambda v: v.center[0])

    # Mark topmost and bottommost vertebrae to exclude from angle calculations
    if len(vertebrae) > 0:
        vertebrae[0].is_end_vertebra = True  # Topmost vertebra

    if len(vertebrae) > 1:
        vertebrae[-1].is_end_vertebra = True  # Bottommost vertebra

    # Identify outlier vertebrae by fitting a curve to the spine
    if len(vertebrae) > 4:
        vertebrae, poly_coeffs, y_values = identify_outliers(vertebrae)
    else:
        poly_coeffs, y_values = None, None

    # Renumber vertebrae IDs (sequential)
    for i, vertebra in enumerate(vertebrae):
        vertebra.id = i + 1

    return vertebrae, poly_coeffs, y_values


def algorithm2_cobb_angle_measurement(vertebrae):
    """
    Algorithm 2: Calculate Cobb angle between vertebrae with the maximum angle,
    excluding the topmost, bottommost, and outlier vertebrae.
    """
    # Filter vertebrae
    valid_vertebrae = [v for v in vertebrae
                      if v.superior_endplate_slope is not None
                      and v.inferior_endplate_slope is not None
                      and not v.is_end_vertebra
                      and not v.is_outlier]

    if len(valid_vertebrae) < 2:
        return 0, None, None, False

    # Extract slopes M1 (superior) and M2 (inferior)
    M1 = [v.superior_endplate_slope for v in valid_vertebrae]
    M2 = [v.inferior_endplate_slope for v in valid_vertebrae]

    # Initialize variables
    max_cobb_angle = 0
    upper_vertebra_id = None
    lower_vertebra_id = None

    # Iterate through all pairs of vertebrae
    for i in range(len(valid_vertebrae)):
        for j in range(i+1, len(valid_vertebrae)):
            m1 = M1[i]
            m2 = M2[j]

            denominator = 1.0 + (m1 * m2)
            if abs(denominator) < 1e-6:
                continue

            tan_theta = abs((m1 - m2) / denominator)
            theta = math.degrees(math.atan(tan_theta))

            if theta > 60:
                continue

            if theta > max_cobb_angle:
                max_cobb_angle = theta
                upper_vertebra_id = valid_vertebrae[i].id
                lower_vertebra_id = valid_vertebrae[j].id

    # Round to nearest 0.5 degree
    max_cobb_angle = round(max_cobb_angle * 2) / 2

    # Check if scoliosis exists
    scoliosis_exists = max_cobb_angle > 10.0

    return max_cobb_angle, upper_vertebra_id, lower_vertebra_id, scoliosis_exists


def classify_scoliosis(angle: float) -> str:
    """Classify scoliosis (English labels) based on Cobb angle."""
    if angle < 10:
        return "Normal"
    elif angle < 25:
        return "Mild Scoliosis"
    elif angle < 40:
        return "Moderate Scoliosis"
    elif angle < 50:
        return "Severe Scoliosis"
    else:
        return "Very Severe Scoliosis"



def analyze_spine_from_image(image):
    """
    Analyze spine curvature from an X-ray image with green highlighted vertebrae
    using Algorithm 1 and Algorithm 2, ignoring the topmost, bottommost, and
    outlier vertebrae.
    """
    # Extract green mask
    green_mask = extract_green_mask(image)

    # Extract vertebrae using Algorithm 1 and identify outliers
    result = extract_vertebrae(green_mask)
    if len(result) == 3:
        vertebrae, poly_coeffs, y_values = result
    else:
        vertebrae = result
        poly_coeffs, y_values = None, None

    # Default result values
    result = {
        'cobb_angles': [],
        'max_angle': 0,
        'classification': 'Normal',
        'vertebrae': [],
        'confidence': 0,
        'valid': False,
        'spine_curve': {
            'poly_coeffs': poly_coeffs,
            'y_values': y_values.tolist() if y_values is not None else None
        } if poly_coeffs is not None else None
    }

    # Check if enough vertebrae detected
    if len(vertebrae) < 4:
        result['error'] = 'Not enough vertebrae detected for reliable measurement'
        return result

    # Calculate Cobb angle using Algorithm 2
    max_angle, upper_id, lower_id, scoliosis_exists = algorithm2_cobb_angle_measurement(vertebrae)

    if max_angle > 0 and upper_id is not None and lower_id is not None:
        # Find the vertebrae objects
        upper_vertebra = next((v for v in vertebrae if v.id == upper_id), None)
        lower_vertebra = next((v for v in vertebrae if v.id == lower_id), None)

        if upper_vertebra and lower_vertebra:
            # Create angle data
            angle_data = {
                'angle': max_angle,
                'vertebra1': upper_id,
                'vertebra2': lower_id,
                'vertebra1_center': (upper_vertebra.center[1], upper_vertebra.center[0]),
                'vertebra2_center': (lower_vertebra.center[1], lower_vertebra.center[0]),
                'vertebra1_angle': math.degrees(math.atan(upper_vertebra.superior_endplate_slope)),
                'vertebra2_angle': math.degrees(math.atan(lower_vertebra.inferior_endplate_slope))
            }

            # Calculate result confidence
            vertebra_count_factor = min(1.0, len(vertebrae) / 8)
            angle_confidence = min(1.0, max_angle / 15) if max_angle > 0 else 0.5

            # Total confidence score
            confidence = 0.9  # High confidence with the algorithm approach
            confidence *= vertebra_count_factor * angle_confidence

            # Update results
            result['cobb_angles'] = [angle_data]
            result['max_angle'] = max_angle
            result['classification'] = classify_scoliosis(max_angle)
            result['confidence'] = confidence
            result['valid'] = True if scoliosis_exists else False

    # Prepare vertebrae data for visualization
    vertebrae_data = []
    for v in vertebrae:
        angle = 0
        if v.superior_endplate_slope is not None:
            angle = math.degrees(math.atan(v.superior_endplate_slope))
        vertebrae_data.append({
            'id': v.id,
            'center': (v.center[1], v.center[0]),
            'angle': angle,
            'bbox': (v.bbox[1], v.bbox[0], v.bbox[3], v.bbox[2]),
            'is_end_vertebra': v.is_end_vertebra,
            'is_outlier': v.is_outlier,
            'distance_from_curve': v.distance_from_curve
        })
    result['vertebrae'] = vertebrae_data
    return result


def _draw_angle_arm(img, center_xy, angle_deg, length, color=(0, 0, 255), thick=6):
    """Line through vertebral center with given orientation (anti-aliased)."""
    cx, cy = int(center_xy[0]), int(center_xy[1])
    rad = np.deg2rad(angle_deg)
    x1 = int(cx - length * np.cos(rad))
    y1 = int(cy - length * np.sin(rad))
    x2 = int(cx + length * np.cos(rad))
    y2 = int(cy + length * np.sin(rad))
    cv2.line(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)


def _put_label(img, text, org, font, scale, color, thick, pad=6, alpha=0.45):
    """
    Text with soft shadow + translucent background.
    Returns ((x1,y1,x2,y2), (tw,th,base)) for downstream placement.
    """
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x0, y0 = org
    x1, y1 = x0 - pad, y0 - th - pad
    x2, y2 = x0 + tw + pad, y0 + base + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # shadow + text
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)
    return (x1, y1, x2, y2), (tw, th, base)


def _label_with_degree(img, text_wo_deg, org, font, scale, color, thick,
                       pad=6, alpha=0.45, deg_radius=3, deg_dx=2, deg_dy=2):
    """
    Writes e.g. '32.0' then draws a small superscript circle as degree sign.
    Avoids unsupported '°' glyph → no more '??'.
    """
    _, (tw, th, _base) = _put_label(img, text_wo_deg, org, font, scale, color, thick, pad, alpha)
    cx = org[0] + tw + deg_dx                 # to the right of the text
    cy = org[1] - th + deg_dy                 # superscript position
    cv2.circle(img, (int(cx), int(cy)), int(deg_radius), color, -1, cv2.LINE_AA)

# --- main visualization ----------------------------------------------------

def visualize_results(image, analysis_results):
    """Visualize analysis results on original image with fixed sizes (no change)."""
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    H, W = vis_image.shape[:2]

    # dynamic scale (kept as before; you asked not to change sizes)
    base_h = 1024.0
    sf = max(0.8, min(3.0, H / base_h))

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    COL_TEXT  = (0, 220, 255)
    COL_CURVE = (255, 255, 0)
    COL_ANGLE = (0, 0, 255)
    COL_IN    = (40, 220, 40)
    COL_END   = (0, 165, 255)
    COL_OUT   = (0, 0, 255)

    fs_small  = 0.6 * sf
    fs_med    = 0.95 * sf
    fs_big    = 1.15 * sf
    th_text   = max(1, int(round(2 * sf)))
    th_curve  = max(2, int(round(3 * sf)))
    th_angle  = max(3, int(round(6 * sf)))
    r_dot     = max(4, int(round(5 * sf)))
    line_len  = max(H, W) // 3
    deg_r     = max(2, int(round(3 * sf)))  # degree-dot size

    # spine curve
    sc = analysis_results.get('spine_curve', None)
    if sc is not None and sc.get('y_values') is not None:
        poly = np.array(sc.get('poly_coeffs'))
        ys = np.array(sc.get('y_values'))
        if poly is not None and len(poly) > 0:
            f = np.poly1d(poly)
            xs = f(ys)
            pts = np.column_stack((xs.astype(np.int32), ys.astype(np.int32))).reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], False, COL_CURVE, th_curve, cv2.LINE_AA)

    # vertebrae dots + ids
    for v in analysis_results.get('vertebrae', []):
        c = tuple(int(a) for a in v['center'])
        if v.get('is_outlier', False):
            color = COL_OUT
        elif v.get('is_end_vertebra', False):
            color = COL_END
        else:
            color = COL_IN
        cv2.circle(vis_image, c, r_dot, color, -1, cv2.LINE_AA)
        cv2.putText(vis_image, str(v['id']), (c[0] + int(8 * sf), c[1]),
                    FONT, max(0.45, 0.45 * sf), (255, 255, 255), 1, cv2.LINE_AA)

    # Cobb arms + mid-text (with degree dot)
    for ad in analysis_results.get('cobb_angles', []):
        if ad['angle'] == 0:
            continue
        v1c = tuple(int(a) for a in ad['vertebra1_center'])
        v2c = tuple(int(a) for a in ad['vertebra2_center'])
        _draw_angle_arm(vis_image, v1c, ad['vertebra1_angle'], line_len, COL_ANGLE, th_angle)
        _draw_angle_arm(vis_image, v2c, ad['vertebra2_angle'], line_len, COL_ANGLE, th_angle)
        cv2.line(vis_image, v1c, v2c, COL_ANGLE, max(2, th_angle - 3), cv2.LINE_AA)

        mid_x = (v1c[0] + v2c[0]) // 2
        mid_y = (v1c[1] + v2c[1]) // 2
        _label_with_degree(
            vis_image, f"{ad['angle']:.1f}",
            (mid_x + int(18 * sf), mid_y),
            FONT, fs_med, COL_ANGLE, th_text,
            pad=max(4, int(6 * sf)), deg_radius=deg_r, deg_dx=max(2, int(2 * sf)), deg_dy=max(2, int(2 * sf))
        )

    # header (with degree dot), classification & legend
    max_angle = float(analysis_results.get('max_angle', 0.0))
    classification = analysis_results.get('classification', 'Normal')
    y0 = int(36 * sf)

    _label_with_degree(
        vis_image, f"Max Cobb Angle: {max_angle:.1f}",
        (int(28 * sf), y0),
        FONT, fs_big, COL_TEXT, th_text,
        pad=max(6, int(8 * sf)), deg_radius=deg_r, deg_dx=max(3, int(3 * sf)), deg_dy=max(3, int(3 * sf))
    )
    _put_label(vis_image, f"Classification: {classification}",
               (int(28 * sf), y0 + int(34 * sf)), FONT, fs_med, COL_TEXT, th_text, pad=max(6, int(8 * sf)))
    _put_label(vis_image, "Excluded: end vertebrae & distant outliers",
               (int(28 * sf), y0 + int(64 * sf)), FONT, fs_small, (0, 200, 255), th_text, pad=max(5, int(7 * sf)))

    return vis_image