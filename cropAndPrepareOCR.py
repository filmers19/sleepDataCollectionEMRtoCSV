from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_a)), int(round(width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_a)), int(round(height_b)))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def preprocess_for_document_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=45, sigmaY=45)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.GaussianBlur(normalized, (5, 5), 0)

    otsu_threshold, binary = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return normalized, binary, int(round(otsu_threshold))


def detect_document(image: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized, binary, otsu_threshold = preprocess_for_document_mask(gray)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy(), {
            "method": "none",
            "otsu_threshold": otsu_threshold,
            "polygon_found": False,
            "used_fallback_bbox": False,
        }

    image_area = image.shape[0] * image.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= image_area * 0.12:
            document_contour = contour
            break
    if document_contour is None:
        document_contour = contours[0]

    peri = cv2.arcLength(document_contour, True)
    approx = cv2.approxPolyDP(document_contour, 0.02 * peri, True)

    if len(approx) == 4:
        warped = four_point_transform(image, approx.reshape(4, 2))
        return warped, {
            "method": "perspective_transform",
            "otsu_threshold": otsu_threshold,
            "polygon_found": True,
            "used_fallback_bbox": False,
            "contour_area": float(cv2.contourArea(document_contour)),
            "points": approx.reshape(4, 2).tolist(),
        }

    x, y, w, h = cv2.boundingRect(document_contour)
    cropped = image[y : y + h, x : x + w].copy()
    return cropped, {
        "method": "bounding_box_fallback",
        "otsu_threshold": otsu_threshold,
        "polygon_found": False,
        "used_fallback_bbox": True,
        "contour_area": float(cv2.contourArea(document_contour)),
        "bbox": [int(x), int(y), int(w), int(h)],
    }


def trim_white_margins(
    image: np.ndarray,
    white_threshold: int = 245,
    min_content_pixels_ratio: float = 0.001,
    margin: int = 8,
) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    non_white = gray < white_threshold
    min_row_pixels = max(5, int(round(width * min_content_pixels_ratio)))
    min_col_pixels = max(5, int(round(height * min_content_pixels_ratio)))

    rows = np.where(non_white.sum(axis=1) >= min_row_pixels)[0]
    cols = np.where(non_white.sum(axis=0) >= min_col_pixels)[0]
    if len(rows) == 0 or len(cols) == 0:
        return image.copy(), {
            "method": "trim_white_margins_none",
            "white_threshold": white_threshold,
            "margin": margin,
            "min_row_pixels": min_row_pixels,
            "min_col_pixels": min_col_pixels,
            "used_fallback_bbox": False,
        }

    y0 = max(0, int(rows[0]) - margin)
    y1 = min(height - 1, int(rows[-1]) + margin)
    x0 = max(0, int(cols[0]) - margin)
    x1 = min(width - 1, int(cols[-1]) + margin)

    cropped = image[y0 : y1 + 1, x0 : x1 + 1].copy()
    return cropped, {
        "method": "trim_white_margins",
        "white_threshold": white_threshold,
        "margin": margin,
        "min_row_pixels": min_row_pixels,
        "min_col_pixels": min_col_pixels,
        "used_fallback_bbox": False,
        "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
    }


def prepare_for_ocr(
    image: np.ndarray,
    scale: float = 3.0,
    denoise: bool = True,
    sharpen: bool = False,
    threshold_method: str = "adaptive",
    clahe: bool = True,
) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast
    if clahe:
        clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe_filter.apply(gray)
    else:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if sharpen:
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        gray = cv2.filter2D(gray, -1, kernel)

    if threshold_method == "adaptive":
        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
    elif threshold_method == "otsu":
        _, result = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    elif threshold_method == "none":
        result = gray
    else:
        raise ValueError(f"Unsupported threshold method: {threshold_method}")

    return result, {
        "method": "prepare_for_ocr_no_upscale",
        "scale": 1.0,
        "scale_requested": scale,
        "denoise": denoise,
        "sharpen": sharpen,
        "threshold_method": threshold_method,
        "clahe": clahe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-mask", default=None)

    parser.add_argument(
        "--mode",
        choices=["document", "trim_whitespace"],
        default="document",
    )
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--min-content-pixels-ratio", type=float, default=0.001)
    parser.add_argument("--margin", type=int, default=8)

    # OCR prep flags
    parser.add_argument("--ocr-ready", action="store_true")
    parser.add_argument("--ocr-scale", type=float, default=3.0)
    parser.add_argument(
        "--threshold-method",
        choices=["adaptive", "otsu", "none"],
        default="adaptive",
    )
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--trim-after-document", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {input_path}")

    if args.mode == "trim_whitespace":
        processed, meta = trim_white_margins(
            image,
            white_threshold=args.white_threshold,
            min_content_pixels_ratio=args.min_content_pixels_ratio,
            margin=args.margin,
        )
    else:
        processed, meta = detect_document(image)

    if args.trim_after_document and args.mode == "document":
        processed, trim_meta = trim_white_margins(
            processed,
            white_threshold=args.white_threshold,
            min_content_pixels_ratio=args.min_content_pixels_ratio,
            margin=args.margin,
        )
        meta["post_trim"] = trim_meta

    if args.ocr_ready:
        processed, ocr_meta = prepare_for_ocr(
            processed,
            scale=args.ocr_scale,
            denoise=not args.no_denoise,
            sharpen=args.sharpen,
            threshold_method=args.threshold_method,
            clahe=not args.no_clahe,
        )
        meta["ocr"] = ocr_meta

    ok = cv2.imwrite(str(output_path), processed)
    if not ok:
        raise RuntimeError(f"Failed to save output image: {output_path}")

    if args.debug_mask:
        debug_path = Path(args.debug_mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if args.mode == "trim_whitespace":
            binary = np.where(gray < args.white_threshold, 255, 0).astype(np.uint8)
        else:
            _, binary, _ = preprocess_for_document_mask(gray)
        cv2.imwrite(str(debug_path), binary)

    print(
        {
            "input": str(input_path),
            "output": str(output_path),
            "original_shape": list(image.shape),
            "processed_shape": list(processed.shape),
            **meta,
        }
    )


if __name__ == "__main__":
    main()
