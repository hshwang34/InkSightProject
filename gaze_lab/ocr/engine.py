"""
OCR engine with fallback support for pytesseract and easyocr.

Provides a unified interface for text recognition with automatic fallback
between different OCR engines based on availability.
"""

import re
from typing import Optional, Tuple

import numpy as np

from ..logging_setup import get_logger

logger = get_logger(__name__)

# Try to import OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

# Global EasyOCR reader cache
_easyocr_readers = {}


def recognize_text(
    image: np.ndarray,
    lang: str = "eng",
    engine_preference: Tuple[str, ...] = ("tesseract", "easyocr"),
    tesseract_config: Optional[str] = None,
) -> str:
    """
    Recognize text from an image using available OCR engines.
    
    Args:
        image: Input image as numpy array (BGR or grayscale)
        lang: Language code(s) for OCR (default: "eng")
        engine_preference: Tuple of engine names to try in order
        tesseract_config: Custom Tesseract configuration string
        
    Returns:
        Extracted text with normalized whitespace
        
    Raises:
        RuntimeError: If no OCR engines are available
    """
    if not TESSERACT_AVAILABLE and not EASYOCR_AVAILABLE:
        raise RuntimeError(
            "No OCR engines available. Install one of the following:\n"
            "  pip install pytesseract  # Requires Tesseract binary\n"
            "  pip install easyocr       # Self-contained but larger\n\n"
            "For Tesseract installation:\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS: brew install tesseract\n"
            "  Linux: sudo apt-get install tesseract-ocr"
        )
    
    # Try engines in preference order
    for engine in engine_preference:
        try:
            if engine == "tesseract" and TESSERACT_AVAILABLE:
                return _recognize_with_tesseract(image, lang, tesseract_config)
            elif engine == "easyocr" and EASYOCR_AVAILABLE:
                return _recognize_with_easyocr(image, lang)
        except Exception as e:
            logger.warning(f"OCR engine '{engine}' failed: {e}")
            continue
    
    # If we get here, all preferred engines failed
    available_engines = []
    if TESSERACT_AVAILABLE:
        available_engines.append("tesseract")
    if EASYOCR_AVAILABLE:
        available_engines.append("easyocr")
    
    raise RuntimeError(
        f"All OCR engines failed. Available engines: {available_engines}"
    )


def _recognize_with_tesseract(
    image: np.ndarray, 
    lang: str, 
    config: Optional[str]
) -> str:
    """Recognize text using Tesseract OCR."""
    try:
        # Default config optimized for UI text and documents
        if config is None:
            config = "--oem 1 --psm 6"
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            import cv2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Run Tesseract
        text = pytesseract.image_to_string(
            image_rgb, 
            lang=lang, 
            config=config
        )
        
        return _normalize_text(text)
        
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract executable not found. Please install Tesseract:\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS: brew install tesseract\n"
            "  Linux: sudo apt-get install tesseract-ocr\n"
            "Ensure tesseract is in your PATH."
        )
    except Exception as e:
        raise RuntimeError(f"Tesseract OCR failed: {e}")


def _recognize_with_easyocr(image: np.ndarray, lang: str) -> str:
    """Recognize text using EasyOCR."""
    try:
        # Convert language code for EasyOCR
        easyocr_lang = _convert_lang_code(lang)
        
        # Get or create reader for this language
        lang_key = tuple(sorted(easyocr_lang))
        if lang_key not in _easyocr_readers:
            logger.info(f"Initializing EasyOCR reader for languages: {easyocr_lang}")
            _easyocr_readers[lang_key] = easyocr.Reader(easyocr_lang, gpu=False)
        
        reader = _easyocr_readers[lang_key]
        
        # Run EasyOCR
        results = reader.readtext(image, detail=0)  # detail=0 returns only text
        
        # Combine all detected text
        text = "\n".join(results)
        
        return _normalize_text(text)
        
    except Exception as e:
        raise RuntimeError(f"EasyOCR failed: {e}")


def _convert_lang_code(lang: str) -> list:
    """Convert Tesseract language codes to EasyOCR format."""
    # Handle common language codes
    lang_mapping = {
        "eng": ["en"],
        "spa": ["es"],
        "fra": ["fr"],
        "deu": ["de"],
        "ita": ["it"],
        "por": ["pt"],
        "rus": ["ru"],
        "chi_sim": ["ch_sim"],
        "chi_tra": ["ch_tra"],
        "jpn": ["ja"],
        "kor": ["ko"],
    }
    
    # Handle multiple languages (e.g., "eng+spa")
    if "+" in lang:
        langs = lang.split("+")
        result = []
        for l in langs:
            result.extend(lang_mapping.get(l, [l]))
        return result
    
    return lang_mapping.get(lang, [lang])


def _normalize_text(text: str) -> str:
    """Normalize extracted text by cleaning whitespace."""
    if not text:
        return ""
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Collapse multiple spaces but preserve line breaks
    text = re.sub(r"[ \t]+", " ", text)
    
    # Remove excessive line breaks (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text


def get_available_engines() -> list:
    """Get list of available OCR engines."""
    engines = []
    if TESSERACT_AVAILABLE:
        engines.append("tesseract")
    if EASYOCR_AVAILABLE:
        engines.append("easyocr")
    return engines


def is_ocr_available() -> bool:
    """Check if any OCR engine is available."""
    return TESSERACT_AVAILABLE or EASYOCR_AVAILABLE
