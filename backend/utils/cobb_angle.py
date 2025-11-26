"""
Improved Cobb Angle Measurement Module for SpineCheck-AI
=========================================================
Enhanced version with robust handling of imperfect segmentations:
- ROI detection to focus on spine region
- Morphological operations to separate merged vertebrae
- Artifact removal using geometric constraints
- Vertebra splitting for merged components
- No artificial zeroing for mild curvatures
"""

import numpy as np
import cv2
from scipy import stats, ndimage
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

# ============================================================================
# Configuration Parameters
# ============================================================================

# Classification thresholds (in degrees)
NORMAL_THRESHOLD = 10.0
MILD_THRESHOLD = 25.0
MODERATE_THRESHOLD = 40.0
SEVERE_THRESHOLD = 60.0

# Algorithm parameters
MAX_ENDPLATE_ANGLE = 35.0
MAX_COBB_ANGLE = 80.0
MIN_VERTEBRA_SEPARATION = 2
MAX_VERTEBRA_SEPARATION = 10
OUTLIER_STD_MULTIPLIER = 2.5
NUM_ENDPLATE_SAMPLES = 20
MIN_VERTEBRA_AREA = 100
MAX_VERTEBRA_AREA = 100000

# New parameters for improved robustness
MIN_VERTEBRA_HEIGHT = 10
MIN_VERTEBRA_WIDTH = 15
MAX_ASPECT_RATIO = 6.0
ROI_WIDTH_FACTOR = 0.5
VERTEBRA_MERGE_THRESHOLD = 3.0
EROSION_ITERATIONS = 1
MIN_VERTEBRA_SOLIDITY = 0.25


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class VertebraInfo:
    """Information about a single vertebra"""
    id: int  # Vertebra index (1-based, from top to bottom)
    mask: np.ndarray  # Binary mask for this vertebra
    center: Tuple[float, float]  # (y, x) centroid coordinates

    # Endplate geometry
    superior_endplate_points: Optional[np.ndarray] = None  # (N, 2) array of (x, y) points
    inferior_endplate_points: Optional[np.ndarray] = None  # (N, 2) array of (x, y) points
    superior_endplate_angle: Optional[float] = None  # degrees
    inferior_endplate_angle: Optional[float] = None  # degrees
    superior_endplate_slope: Optional[float] = None  # tan(angle)
    inferior_endplate_slope: Optional[float] = None  # tan(angle)

    # Status flags
    is_end_vertebra: bool = False  # First or last vertebra
    is_outlier: bool = False  # Too far from spine centerline

    # Minimum enclosing rectangle
    min_rect: Optional[Tuple] = None  # Output from cv2.minAreaRect

    # Additional properties for robustness
    area: int = 0
    solidity: float = 0.0
    aspect_ratio: float = 0.0

    def __repr__(self):
        sup_ang = f"{self.superior_endplate_angle:.1f}" if self.superior_endplate_angle is not None else "None"
        inf_ang = f"{self.inferior_endplate_angle:.1f}" if self.inferior_endplate_angle is not None else "None"
        return f"Vertebra(id={self.id}, center={self.center}, sup={sup_ang}°, inf={inf_ang}°, area={self.area})"


# ============================================================================
# Helper Functions
# ============================================================================

def detect_spine_roi(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detect the region of interest containing the spine.

    This helps exclude artifacts on the sides (ribs, scale markers, etc.)
    by focusing on the central vertical column where the spine is located.

    Args:
        mask: Binary mask of all detected regions

    Returns:
        (x_min, x_max, y_min, y_max) defining the ROI
    """
    h, w = mask.shape

    # Find all non-zero points
    points = np.column_stack(np.where(mask > 0))

    if len(points) == 0:
        # Return full image if no points found
        return 0, w, 0, h

    # Calculate vertical projection (sum along rows)
    vertical_projection = np.sum(mask, axis=0)

    # Find the main vertical region with smoothing
    smoothed_projection = ndimage.gaussian_filter1d(vertical_projection.astype(float), sigma=5)

    # Find peak and determine ROI width
    peak_x = np.argmax(smoothed_projection)
    roi_half_width = int(w * ROI_WIDTH_FACTOR / 2)

    # Calculate ROI bounds
    x_min = max(0, peak_x - roi_half_width)
    x_max = min(w, peak_x + roi_half_width)

    # For y bounds, use the actual extent of the mask within the x ROI
    roi_mask = mask[:, x_min:x_max]
    y_coords = np.where(np.any(roi_mask > 0, axis=1))[0]

    if len(y_coords) > 0:
        y_min = y_coords[0]
        y_max = y_coords[-1] + 1
    else:
        y_min, y_max = 0, h

    return x_min, x_max, y_min, y_max


def apply_roi_to_mask(mask: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Apply ROI to mask, zeroing out regions outside the ROI.

    Args:
        mask: Binary mask
        roi: (x_min, x_max, y_min, y_max) ROI bounds

    Returns:
        Masked image with regions outside ROI set to zero
    """
    x_min, x_max, y_min, y_max = roi
    roi_mask = np.zeros_like(mask)
    roi_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
    return roi_mask


def separate_merged_vertebrae(mask: np.ndarray) -> np.ndarray:
    """
    Attempt to separate merged vertebrae using adaptive morphological operations.
    Uses a gentler approach to avoid breaking apart valid vertebrae.

    Args:
        mask: Binary mask with potentially merged vertebrae

    Returns:
        Processed mask with separated vertebrae
    """
    # First check if separation is needed by analyzing component sizes
    num_labels_orig, labels_orig = cv2.connectedComponents(mask)

    # If we already have a reasonable number of components, be very gentle
    if num_labels_orig > 10:  # Already have many components
        # Just do light opening to clean up connections
        kernel_light = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_light, iterations=1)

    # Apply gradient-based separation for truly merged regions
    # Use distance transform to find centers of merged regions
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Find peaks in the distance transform (vertebra centers)
    # Threshold to get only the strongest peaks
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find unknown region (between foreground and background)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling for watershed
    num_markers, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that background is 1, not 0
    markers = markers + 1

    # Mark the unknown region as 0
    markers[unknown == 255] = 0

    # Apply watershed if we have multiple markers
    if num_markers > 2:  # More than just background
        # Convert mask to 3-channel for watershed
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_3ch, markers)

        # Create result mask
        separated_mask = np.zeros_like(mask)
        separated_mask[markers > 1] = 255

        # Clean up with light morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        separated_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_CLOSE, kernel_small)

        return separated_mask
    else:
        # If watershed didn't help, try gentle erosion
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask, kernel_erode, iterations=EROSION_ITERATIONS)

        # Find connected components in eroded image
        num_labels, labels = cv2.connectedComponents(eroded)

        # If erosion created more components, use it
        if num_labels > num_labels_orig + 2:
            # Dilate each component separately
            separated_mask = np.zeros_like(mask)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            for label_id in range(1, num_labels):
                component = (labels == label_id).astype(np.uint8) * 255
                # Dilate back carefully
                dilated = cv2.dilate(component, kernel_dilate, iterations=EROSION_ITERATIONS)
                # Only keep the dilation within original mask bounds
                dilated = cv2.bitwise_and(dilated, mask)
                separated_mask = cv2.bitwise_or(separated_mask, dilated)

            return separated_mask
        else:
            # Return original if separation didn't help
            return mask


def split_large_component(component_mask: np.ndarray, max_vertebra_height: int = 80) -> List[np.ndarray]:
    """
    Split a large connected component that likely contains multiple vertebrae.

    Args:
        component_mask: Binary mask of a single large component
        max_vertebra_height: Estimated maximum height of a single vertebra

    Returns:
        List of binary masks for split vertebrae
    """
    # Find the bounding box
    coords = np.column_stack(np.where(component_mask > 0))
    if len(coords) == 0:
        return [component_mask]

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = y_max - y_min

    # If component is not too tall, return as is
    if height <= max_vertebra_height:
        return [component_mask]

    # Estimate number of vertebrae in this component
    num_vertebrae = max(2, min(5, int(height / (max_vertebra_height * 0.7))))

    # Try horizontal cutting with varying intensities
    split_masks = []

    for strength in [0.3, 0.5, 0.7]:
        # Create horizontal cuts at regular intervals
        temp_mask = component_mask.copy()
        cut_height = 3  # Height of each cut

        for i in range(1, num_vertebrae):
            cut_y = y_min + int(i * height / num_vertebrae)
            cut_start = max(0, cut_y - cut_height // 2)
            cut_end = min(temp_mask.shape[0], cut_y + cut_height // 2)

            # Reduce the mask intensity at cut locations
            temp_mask[cut_start:cut_end, :] = (temp_mask[cut_start:cut_end, :] * (1 - strength)).astype(np.uint8)

        # Apply threshold and find new components
        temp_mask = (temp_mask > 128).astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(temp_mask)

        # Collect the split components
        if num_labels > 2:  # More than just background and one component
            split_masks = []
            for label_id in range(1, num_labels):
                split_mask = (labels == label_id).astype(np.uint8) * 255
                # Apply original mask to maintain boundaries
                split_mask = cv2.bitwise_and(split_mask, component_mask)
                if np.sum(split_mask) > MIN_VERTEBRA_AREA * 255:
                    split_masks.append(split_mask)

            if len(split_masks) >= 2:
                return split_masks

    # If splitting failed, return original
    return [component_mask]


def extract_green_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract the green/cyan highlighted vertebrae region from the image.
    Enhanced version with better color detection and noise removal.

    Args:
        image: BGR image with vertebrae highlighted in green/cyan

    Returns:
        Binary mask (0 or 255) of the highlighted region
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green/cyan colors in HSV (broader range)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    # Create mask for green regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Also check for bright green in BGR space
    b, g, r = cv2.split(image)
    bright_green = (g > 80) & (g > r * 1.3) & (g > b * 1.3)
    bright_green = bright_green.astype(np.uint8) * 255

    # Combine masks
    combined_mask = cv2.bitwise_or(green_mask, bright_green)

    # Remove small noise with opening
    kernel_small = np.ones((2, 2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)

    # Fill small holes
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return combined_mask


def filter_by_geometry(contour: np.ndarray) -> Tuple[bool, Dict]:
    """
    Check if a contour has the geometric properties of a vertebra.

    Args:
        contour: OpenCV contour

    Returns:
        (is_valid, properties) where properties contains area, solidity, aspect_ratio
    """
    area = cv2.contourArea(contour)

    # Quick area check
    if area < MIN_VERTEBRA_AREA or area > MAX_VERTEBRA_AREA:
        return False, {'area': area, 'solidity': 0, 'aspect_ratio': 0}

    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Check dimensions
    if h < MIN_VERTEBRA_HEIGHT or w < MIN_VERTEBRA_WIDTH:
        return False, {'area': area, 'solidity': 0, 'aspect_ratio': 0}

    # Calculate aspect ratio
    aspect_ratio = max(w/h, h/w)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, {'area': area, 'solidity': 0, 'aspect_ratio': aspect_ratio}

    # Calculate solidity (area / convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    if solidity < MIN_VERTEBRA_SOLIDITY:
        return False, {'area': area, 'solidity': solidity, 'aspect_ratio': aspect_ratio}

    return True, {'area': area, 'solidity': solidity, 'aspect_ratio': aspect_ratio}


def extract_vertebrae(mask: np.ndarray) -> List[VertebraInfo]:
    """
    Enhanced vertebra extraction with ROI detection and better filtering.

    Args:
        mask: Binary mask with vertebrae highlighted (0 or 255)

    Returns:
        List of VertebraInfo objects sorted from top to bottom
    """
    # Step 1: Detect and apply ROI to focus on spine region
    roi = detect_spine_roi(mask)
    mask_roi = apply_roi_to_mask(mask, roi)

    # Step 2: Separate merged vertebrae
    separated_mask = separate_merged_vertebrae(mask_roi)

    # Step 3: Find connected components
    contours, _ = cv2.findContours(separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertebrae = []

    for contour in contours:
        # Filter by geometry
        is_valid, props = filter_by_geometry(contour)
        if not is_valid:
            continue

        # Create individual mask for this vertebra
        vert_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(vert_mask, [contour], -1, 255, -1)

        # Check if this component might be multiple merged vertebrae
        x, y, w, h = cv2.boundingRect(contour)
        if h > w * VERTEBRA_MERGE_THRESHOLD and props['area'] > 3000:
            # Try to split it
            split_masks = split_large_component(vert_mask)

            for split_mask in split_masks:
                # Find contour of split mask
                split_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(split_contours) == 0:
                    continue

                split_contour = max(split_contours, key=cv2.contourArea)
                is_valid, split_props = filter_by_geometry(split_contour)

                if is_valid:
                    # Calculate centroid
                    M = cv2.moments(split_contour)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]

                        vert_info = VertebraInfo(
                            id=0,  # Will be assigned later
                            mask=split_mask,
                            center=(cy, cx),
                            area=split_props['area'],
                            solidity=split_props['solidity'],
                            aspect_ratio=split_props['aspect_ratio']
                        )
                        vertebrae.append(vert_info)
        else:
            # Use original component
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                vert_info = VertebraInfo(
                    id=0,  # Will be assigned later
                    mask=vert_mask,
                    center=(cy, cx),
                    area=props['area'],
                    solidity=props['solidity'],
                    aspect_ratio=props['aspect_ratio']
                )
                vertebrae.append(vert_info)

    # Sort by y-coordinate (top to bottom)
    vertebrae.sort(key=lambda v: v.center[0])

    # Assign IDs
    for i, vert in enumerate(vertebrae):
        vert.id = i + 1

    # Mark end vertebrae (first and last)
    if len(vertebrae) > 0:
        vertebrae[0].is_end_vertebra = True
        if len(vertebrae) > 1:
            vertebrae[-1].is_end_vertebra = True

    return vertebrae


def normalize_angle(angle_deg: float) -> float:
    """Normalize angle to range (-90, 90] degrees."""
    angle_deg = angle_deg % 360
    if angle_deg > 180:
        angle_deg -= 360

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg <= -90:
        angle_deg += 180

    return angle_deg


def fit_robust_line(points: np.ndarray) -> Tuple[float, float]:
    """Fit a robust line using Theil-Sen regression."""
    if len(points) < 2:
        return 0.0, 0.0

    x = points[:, 0]
    y = points[:, 1]

    if np.std(x) < 1e-6:
        return 1e6, np.mean(y) if len(y) > 0 else 0.0

    try:
        slope, intercept, _, _ = stats.theilslopes(y, x)
        return slope, intercept
    except Exception:
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            return slope, intercept
        return 0.0, np.mean(y) if len(y) > 0 else 0.0


def sample_edge_points(rect_points: np.ndarray, num_samples: int = NUM_ENDPLATE_SAMPLES) -> np.ndarray:
    """Sample points uniformly along an edge."""
    p1, p2 = rect_points[0], rect_points[1]
    t = np.linspace(0, 1, num_samples)

    points = np.zeros((num_samples, 2))
    points[:, 0] = p1[0] + t * (p2[0] - p1[0])
    points[:, 1] = p1[1] + t * (p2[1] - p1[1])

    return points


def compute_acute_angle_difference(angle1_deg: float, angle2_deg: float) -> float:
    """Compute the acute angle difference between two angles in degrees."""
    diff = abs(angle1_deg - angle2_deg) % 180
    return min(diff, 180 - diff)


def fit_spine_centerline(vertebrae: List[VertebraInfo], degree: int = 3) -> Tuple[np.poly1d, float]:
    """Fit a polynomial curve through vertebra centers."""
    valid_vertebrae = [v for v in vertebrae if not v.is_outlier]

    if len(valid_vertebrae) < 2:
        return np.poly1d([0]), 0.0

    x_coords = np.array([v.center[1] for v in valid_vertebrae])
    y_coords = np.array([v.center[0] for v in valid_vertebrae])

    try:
        degree = min(degree, len(valid_vertebrae) - 1)
        coeffs = np.polyfit(x_coords, y_coords, degree)
        poly_func = np.poly1d(coeffs)

        y_pred = poly_func(x_coords)
        residuals = y_coords - y_pred
        rmse = np.sqrt(np.mean(residuals**2))

        return poly_func, rmse
    except Exception as e:
        warnings.warn(f"Failed to fit spine centerline: {e}")
        return np.poly1d([0]), 0.0


def mark_outliers(vertebrae: List[VertebraInfo]) -> None:
    """Mark vertebrae that are outliers based on centerline fit."""
    if len(vertebrae) < 3:
        return

    poly_func, _ = fit_spine_centerline(vertebrae, degree=2)

    distances = []
    for vert in vertebrae:
        if not vert.is_end_vertebra:
            expected_y = poly_func(vert.center[1])
            distance = abs(vert.center[0] - expected_y)
            distances.append(distance)

    if len(distances) > 0:
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + OUTLIER_STD_MULTIPLIER * std_dist

        idx = 0
        for vert in vertebrae:
            if not vert.is_end_vertebra:
                expected_y = poly_func(vert.center[1])
                distance = abs(vert.center[0] - expected_y)
                if distance > threshold:
                    vert.is_outlier = True
                idx += 1


# ============================================================================
# Core Algorithms
# ============================================================================

def algorithm1_minimum_enclosing_rectangle(vertebra: VertebraInfo) -> None:
    """
    Apply Algorithm 1: Minimum enclosing rectangle and endplate approximation.
    """
    points = np.column_stack(np.where(vertebra.mask > 0))
    points_xy = points[:, [1, 0]]

    if len(points_xy) < 5:
        return

    min_rect = cv2.minAreaRect(points_xy.astype(np.float32))
    vertebra.min_rect = min_rect

    box = cv2.boxPoints(min_rect)
    box = np.int32(box)

    # Determine top and bottom edges
    sorted_by_y = sorted(box, key=lambda p: p[1])
    top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_points = sorted(sorted_by_y[2:], key=lambda p: p[0])

    # Sample points along edges
    superior_points = sample_edge_points(np.array(top_points))
    inferior_points = sample_edge_points(np.array(bottom_points))

    # Fit lines to endplates
    sup_slope, sup_intercept = fit_robust_line(superior_points)
    inf_slope, inf_intercept = fit_robust_line(inferior_points)

    # Convert to angles
    sup_angle = np.degrees(np.arctan(sup_slope))
    inf_angle = np.degrees(np.arctan(inf_slope))

    # Normalize angles
    sup_angle = normalize_angle(sup_angle)
    inf_angle = normalize_angle(inf_angle)

    # Validate endplate angles
    if abs(sup_angle) <= MAX_ENDPLATE_ANGLE:
        vertebra.superior_endplate_points = superior_points
        vertebra.superior_endplate_angle = sup_angle
        vertebra.superior_endplate_slope = sup_slope

    if abs(inf_angle) <= MAX_ENDPLATE_ANGLE:
        vertebra.inferior_endplate_points = inferior_points
        vertebra.inferior_endplate_angle = inf_angle
        vertebra.inferior_endplate_slope = inf_slope


def algorithm2_cobb_angle_measurement(vertebrae: List[VertebraInfo]) -> Dict:
    """
    Apply Algorithm 2: Cobb angle measurement with improved selection.
    REMOVED: Straight spine detection that artificially sets angle to 0
    """
    valid_vertebrae = [
        v for v in vertebrae
        if not v.is_end_vertebra
        and not v.is_outlier
        and v.superior_endplate_angle is not None
        and v.inferior_endplate_angle is not None
    ]

    if len(valid_vertebrae) < 2:
        return {
            'valid': False,
            'max_angle': 0.0,
            'superior_vertebra': None,
            'inferior_vertebra': None
        }

    # Find the pair with maximum Cobb angle
    max_angle = 0.0
    best_pair = None

    for i in range(len(valid_vertebrae)):
        for j in range(i + MIN_VERTEBRA_SEPARATION,
                       min(len(valid_vertebrae), i + MAX_VERTEBRA_SEPARATION + 1)):
            v1 = valid_vertebrae[i]
            v2 = valid_vertebrae[j]

            # Try all four combinations
            angles = [
                compute_acute_angle_difference(v1.superior_endplate_angle, v2.inferior_endplate_angle),
                compute_acute_angle_difference(v1.inferior_endplate_angle, v2.superior_endplate_angle),
                compute_acute_angle_difference(v1.superior_endplate_angle, v2.superior_endplate_angle),
                compute_acute_angle_difference(v1.inferior_endplate_angle, v2.inferior_endplate_angle)
            ]

            for k, angle in enumerate(angles):
                if angle > max_angle and angle <= MAX_COBB_ANGLE:
                    max_angle = angle

                    if k == 0:
                        endplate1, endplate2 = 'superior', 'inferior'
                        angle1, angle2 = v1.superior_endplate_angle, v2.inferior_endplate_angle
                    elif k == 1:
                        endplate1, endplate2 = 'inferior', 'superior'
                        angle1, angle2 = v1.inferior_endplate_angle, v2.superior_endplate_angle
                    elif k == 2:
                        endplate1, endplate2 = 'superior', 'superior'
                        angle1, angle2 = v1.superior_endplate_angle, v2.superior_endplate_angle
                    else:
                        endplate1, endplate2 = 'inferior', 'inferior'
                        angle1, angle2 = v1.inferior_endplate_angle, v2.inferior_endplate_angle

                    best_pair = {
                        'superior_vertebra': v1.id,
                        'inferior_vertebra': v2.id,
                        'superior_endplate': endplate1,
                        'inferior_endplate': endplate2,
                        'superior_angle': angle1,
                        'inferior_angle': angle2,
                        'superior_center': v1.center,
                        'inferior_center': v2.center
                    }

    if best_pair is None:
        return {
            'valid': False,
            'max_angle': 0.0,
            'superior_vertebra': None,
            'inferior_vertebra': None
        }

    result = {
        'valid': True,
        'max_angle': max_angle,
        **best_pair
    }

    return result


def classify_scoliosis(cobb_angle: float) -> str:
    """Classify scoliosis severity based on Cobb angle."""
    if cobb_angle < NORMAL_THRESHOLD:
        return "Normal"
    elif cobb_angle < MILD_THRESHOLD:
        return "Mild"
    elif cobb_angle < MODERATE_THRESHOLD:
        return "Moderate"
    elif cobb_angle < SEVERE_THRESHOLD:
        return "Severe"
    else:
        return "Very Severe"


# ============================================================================
# Main Entry Point
# ============================================================================

def analyze_spine_from_image(image: np.ndarray) -> Dict:
    """
    Main entry point for analyzing spine curvature from an X-ray image.
    Now with improved robustness to imperfect segmentations.

    Args:
        image: BGR image with vertebrae highlighted in green/cyan

    Returns:
        Dictionary with:
        - max_angle: Maximum Cobb angle in degrees
        - classification: Severity classification
        - valid: Whether measurement is valid
        - cobb_angles: List of angle measurements
        - vertebrae: List of vertebra information
    """
    # Step 1: Extract the green mask
    mask = extract_green_mask(image)

    # Step 2: Extract individual vertebrae with improved processing
    vertebrae = extract_vertebrae(mask)

    if len(vertebrae) == 0:
        return {
            'max_angle': 0.0,
            'classification': 'Normal',
            'valid': False,
            'cobb_angles': [],
            'vertebrae': []
        }

    # Step 3: Apply minimum enclosing rectangle algorithm
    for vert in vertebrae:
        algorithm1_minimum_enclosing_rectangle(vert)

    # Step 4: Mark outliers
    mark_outliers(vertebrae)

    # Step 5: Measure Cobb angle
    cobb_result = algorithm2_cobb_angle_measurement(vertebrae)

    max_angle = cobb_result['max_angle']
    classification = classify_scoliosis(max_angle)

    # Build response
    cobb_angles = []
    if cobb_result['valid'] and cobb_result['superior_vertebra'] is not None:
        sup_center = cobb_result['superior_center']
        inf_center = cobb_result['inferior_center']

        cobb_angles.append({
            'angle': max_angle,
            'vertebra1': cobb_result['superior_vertebra'],
            'vertebra2': cobb_result['inferior_vertebra'],
            'vertebra1_center': (sup_center[1], sup_center[0]),
            'vertebra2_center': (inf_center[1], inf_center[0]),
            'vertebra1_angle': cobb_result['superior_angle'],
            'vertebra2_angle': cobb_result['inferior_angle']
        })

    vertebrae_list = []
    for vert in vertebrae:
        vert_dict = {
            'id': vert.id,
            'center': (vert.center[1], vert.center[0]),
            'angle': vert.superior_endplate_angle if vert.superior_endplate_angle is not None else 0.0,
            'bbox': (
                list(vert.min_rect[0]) + [vert.min_rect[1][0], vert.min_rect[1][1]]
                if vert.min_rect else [0, 0, 0, 0]
            ),
            'is_end': vert.is_end_vertebra,
            'is_outlier': vert.is_outlier,
            'area': vert.area,
            'solidity': vert.solidity,
            'aspect_ratio': vert.aspect_ratio
        }
        vertebrae_list.append(vert_dict)

    return {
        'max_angle': max_angle,
        'classification': classification,
        'valid': cobb_result['valid'],
        'cobb_angles': cobb_angles,
        'vertebrae': vertebrae_list,
        '_vertebrae_objects': vertebrae,  # Internal: for visualization
        '_cobb_result': cobb_result  # Internal: for visualization
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_results(image: np.ndarray, analysis_results: Dict) -> np.ndarray:
    """
    Enhanced visualization showing ROI, vertebrae, and measurements.
    """
    vis_image = image.copy()

    vertebrae_objects = analysis_results.get('_vertebrae_objects', [])
    cobb_result = analysis_results.get('_cobb_result', {})
    max_angle = analysis_results.get('max_angle', 0.0)
    classification = analysis_results.get('classification', 'Normal')

    if len(vertebrae_objects) == 0:
        cv2.putText(
            vis_image, "No vertebrae detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return vis_image

    # Draw spine centerline if possible
    if len(vertebrae_objects) >= 3:
        try:
            poly_func, _ = fit_spine_centerline(vertebrae_objects, degree=3)

            x_coords = [v.center[1] for v in vertebrae_objects]
            x_min, x_max = min(x_coords), max(x_coords)
            x_curve = np.linspace(x_min, x_max, 100)
            y_curve = poly_func(x_curve)

            pts = np.column_stack([x_curve, y_curve]).astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(vis_image, tuple(pts[i]), tuple(pts[i+1]), (255, 255, 0), 2)
        except:
            pass

    # Draw vertebra centers and information
    for vert in vertebrae_objects:
        cx, cy = int(vert.center[1]), int(vert.center[0])

        # Color coding
        if vert.is_end_vertebra:
            color = (0, 165, 255)  # Orange
            radius = 5
        elif vert.is_outlier:
            color = (0, 0, 255)  # Red
            radius = 5
        else:
            color = (0, 255, 0)  # Green
            radius = 4

        cv2.circle(vis_image, (cx, cy), radius, color, -1)
        cv2.circle(vis_image, (cx, cy), radius + 2, (255, 255, 255), 1)

        # Draw ID and area
        text = f"{vert.id}"
        if vert.area > 0:
            text += f" ({int(vert.area)})"
        cv2.putText(
            vis_image, text, (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )

    # Draw endplate lines for selected pair
    if cobb_result.get('valid') and cobb_result.get('superior_vertebra'):
        sup_vert = None
        inf_vert = None
        for v in vertebrae_objects:
            if v.id == cobb_result['superior_vertebra']:
                sup_vert = v
            if v.id == cobb_result['inferior_vertebra']:
                inf_vert = v

        if sup_vert and sup_vert.superior_endplate_points is not None:
            points = sup_vert.superior_endplate_points.astype(np.int32)
            if len(points) >= 2:
                p1, p2 = points[0], points[-1]
                # Extend the line for better visualization
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx, dy = dx/length, dy/length
                    p1_ext = (int(p1[0] - dx * 30), int(p1[1] - dy * 30))
                    p2_ext = (int(p2[0] + dx * 30), int(p2[1] + dy * 30))
                    cv2.line(vis_image, p1_ext, p2_ext, (0, 0, 255), 2)

        if inf_vert and inf_vert.inferior_endplate_points is not None:
            points = inf_vert.inferior_endplate_points.astype(np.int32)
            if len(points) >= 2:
                p1, p2 = points[0], points[-1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx, dy = dx/length, dy/length
                    p1_ext = (int(p1[0] - dx * 30), int(p1[1] - dy * 30))
                    p2_ext = (int(p2[0] + dx * 30), int(p2[1] + dy * 30))
                    cv2.line(vis_image, p1_ext, p2_ext, (0, 0, 255), 2)

    # Add text overlay with shadowed background
    y_offset = 30
    line_height = 35

    # Create semi-transparent overlay for text background
    overlay = vis_image.copy()
    cv2.rectangle(overlay, (5, 5), (350, 120), (0, 0, 0), -1)
    vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)

    # Draw text with shadow effect
    texts = [
        (f"Cobb Angle: {max_angle:.1f}°", (10, y_offset)),
        (f"Classification: {classification}", (10, y_offset + line_height)),
        ("Enhanced detection with ROI & separation", (10, y_offset + line_height * 2))
    ]

    for text, (x, y) in texts:
        # Shadow
        cv2.putText(vis_image, text, (x+1, y+1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # Main text
        cv2.putText(vis_image, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis_image


# ============================================================================
# Backwards Compatibility Function
# ============================================================================

def measure_cobb_angle_from_green_overlay(image_bgr: np.ndarray) -> Dict:
    """
    Backward compatible function name for measuring Cobb angle.

    Args:
        image_bgr: BGR image with green vertebra overlay

    Returns:
        Dictionary with measurement results
    """
    return analyze_spine_from_image(image_bgr)