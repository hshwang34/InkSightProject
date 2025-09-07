"""
Image preprocessing utilities for OCR.

Provides functions to enhance image quality before OCR processing,
including noise reduction, contrast enhancement, and binarization.
"""

import cv2
import numpy as np

from ..logging_setup import get_logger

logger = get_logger(__name__)


def prep_for_ocr(
    image: np.ndarray,
    mode: str = "auto",
    deskew: bool = False,
) -> np.ndarray:
    """
    Preprocess image for optimal OCR results.
    
    Args:
        image: Input image (BGR or grayscale)
        mode: Processing mode - "auto", "doc", or "ui"
            - auto: Pick thresholds based on image characteristics
            - doc: CLAHE + adaptive threshold (good for printed text)
            - ui: Light denoise + gamma/contrast, avoid over-thresholding
        deskew: Whether to attempt deskewing rotated text
        
    Returns:
        Preprocessed image optimized for OCR
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    logger.debug(f"Preprocessing image {gray.shape} with mode '{mode}'")
    
    # Apply mode-specific preprocessing
    if mode == "auto":
        processed = _preprocess_auto(gray)
    elif mode == "doc":
        processed = _preprocess_document(gray)
    elif mode == "ui":
        processed = _preprocess_ui(gray)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    
    # Optional deskewing
    if deskew:
        processed = _deskew_image(processed)
    
    return processed


def _preprocess_auto(image: np.ndarray) -> np.ndarray:
    """Automatically choose preprocessing based on image characteristics."""
    # Analyze image characteristics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # High contrast images (like UI screenshots) - use UI mode
    if std_intensity > 60 and mean_intensity > 100:
        logger.debug("Auto mode: detected UI-like image")
        return _preprocess_ui(image)
    
    # Low contrast or document-like images - use doc mode
    else:
        logger.debug("Auto mode: detected document-like image")
        return _preprocess_document(image)


def _preprocess_document(image: np.ndarray) -> np.ndarray:
    """Preprocess for document/printed text (aggressive enhancement)."""
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Light denoising
    denoised = cv2.medianBlur(enhanced, 3)
    
    # Adaptive thresholding for binarization
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C parameter
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def _preprocess_ui(image: np.ndarray) -> np.ndarray:
    """Preprocess for UI/screen text (preserve details)."""
    # Light gaussian blur to reduce noise
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Gamma correction for better contrast
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(denoised, table)
    
    # Light contrast enhancement
    enhanced = cv2.convertScaleAbs(gamma_corrected, alpha=1.1, beta=10)
    
    # Only apply mild thresholding if image is very low contrast
    if np.std(enhanced) < 30:
        # Use Otsu's method for automatic threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    return enhanced


def _deskew_image(image: np.ndarray) -> np.ndarray:
    """Attempt to deskew rotated text using Hough line detection."""
    try:
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) == 0:
            logger.debug("No lines detected for deskewing")
            return image
        
        # Calculate rotation angle from detected lines
        angles = []
        for rho, theta in lines[:20]:  # Use first 20 lines
            angle = theta - np.pi/2  # Convert to rotation angle
            angles.append(angle)
        
        # Use median angle to avoid outliers
        rotation_angle = np.median(angles) * 180 / np.pi
        
        # Only rotate if angle is significant (> 1 degree)
        if abs(rotation_angle) > 1:
            logger.debug(f"Deskewing by {rotation_angle:.2f} degrees")
            
            # Get image center and rotation matrix
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
        
    except Exception as e:
        logger.warning(f"Deskewing failed: {e}")
        return image


def enhance_contrast(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """
    Enhance image contrast using specified method.
    
    Args:
        image: Input grayscale image
        method: Enhancement method ("clahe", "histogram", "gamma")
    
    Returns:
        Contrast-enhanced image
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    elif method == "histogram":
        return cv2.equalizeHist(image)
    
    elif method == "gamma":
        # Gamma correction with gamma=1.2
        inv_gamma = 1.0 / 1.2
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")


def denoise_image(image: np.ndarray, method: str = "median") -> np.ndarray:
    """
    Remove noise from image.
    
    Args:
        image: Input image
        method: Denoising method ("median", "gaussian", "bilateral")
    
    Returns:
        Denoised image
    """
    if method == "median":
        return cv2.medianBlur(image, 5)
    
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def binarize_image(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert image to binary (black and white).
    
    Args:
        image: Input grayscale image
        method: Binarization method ("adaptive", "otsu", "threshold")
    
    Returns:
        Binary image
    """
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    
    elif method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    elif method == "threshold":
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    else:
        raise ValueError(f"Unknown binarization method: {method}")
