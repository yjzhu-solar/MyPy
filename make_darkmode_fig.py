#!/usr/bin/env python3
"""
make_darkmode_fig.py

Make white (or near-white) background transparent and optionally overlay
white text label for a figure — useful for dark-mode PowerPoint slides.

Supports PNG, JPG, and JPEG input formats. Output is always PNG to support transparency.

Usage:
    python make_darkmode_fig.py --input INPUT_PATH --output OUTPUT_PATH [--label LABEL]
"""

import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.morphology import remove_small_objects
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Convert figure to dark-mode friendly PNG (transparent bg, white font).")
    parser.add_argument('--input', '-i', required=True, help='Path to input image (PNG, JPG, or JPEG).')
    parser.add_argument('--output', '-o', required=True, help='Path for output image (PNG, with transparency).')
    parser.add_argument('--label', '-l', default=None, help='Optional white text label to overlay (e.g., "Panel A").')
    parser.add_argument('--threshold', '-t', type=int, default=245,
                        help='White threshold (0-255) above which pixels are made transparent. Default: 245.')
    parser.add_argument('--black-threshold', '-b', type=int, default=10,
                        help='Black threshold (0-255) below which pixels are made white. Default: 10.')
    parser.add_argument('--min-black-size', '-m', type=int, default=5,
                        help='Minimum size (in pixels) of black regions to convert to white. Smaller regions are ignored as noise. Default: 5.')
    return parser.parse_args()

def main():
    args = parse_args()

    inp = args.input
    outp = args.output
    label = args.label
    thresh = args.threshold
    black_thresh = args.black_threshold
    min_black_size = args.min_black_size

    if not os.path.isfile(inp):
        print(f"Error: input file '{inp}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Check if input format is supported
    _, ext = os.path.splitext(inp.lower())
    supported_formats = ['.png', '.jpg', '.jpeg']
    if ext not in supported_formats:
        print(f"Warning: input file '{inp}' may not be a supported format. Supported: {', '.join(supported_formats)}")
    
    # Ensure output is PNG format for transparency support
    out_base, out_ext = os.path.splitext(outp)
    if out_ext.lower() not in ['.png', '']:
        print(f"Warning: output format changed to PNG (required for transparency). Output will be: {out_base}.png")
        outp = out_base + '.png'

    # Load image and convert to RGBA (works for both PNG and JPG)
    img = Image.open(inp).convert("RGBA")
    data = np.array(img)
    
    # Create mask of near-white pixels (work with original dimensions)
    r = data[..., 0]
    g = data[..., 1]
    b = data[..., 2]
    white_mask = (r > thresh) & (g > thresh) & (b > thresh)
    
    # Make white pixels transparent
    data[..., 3][white_mask] = 0
    
    # Make black pixels white (only if black_thresh > 0)
    if black_thresh > 0:
        # Create mask of near-black pixels
        black_mask = (r < black_thresh) & (g < black_thresh) & (b < black_thresh)
        
        # Filter out small isolated black regions (noise) using skimage morphology
        if min_black_size > 0:
            # Remove small connected components (noise pixels)
            black_mask = remove_small_objects(black_mask, min_size=min_black_size, connectivity=2)
        
        # Make black pixels white (only large regions, not noise)
        data[..., 0][black_mask] = 255  # Red channel
        data[..., 1][black_mask] = 255  # Green channel
        data[..., 2][black_mask] = 255  # Blue channel
    
    img2 = Image.fromarray(data)

    # If label requested, overlay white text
    if label:
        draw = ImageDraw.Draw(img2)
        try:
            font = ImageFont.truetype("arial.ttf", size=24)
        except IOError:
            font = ImageFont.load_default()
        # Position: top-left with some padding
        padding = 20
        draw.text((padding, padding), label, fill=(255,255,255,255), font=font)

    # Save output
    img2.save(outp, format="PNG")
    print(f"Saved dark-mode version to '{outp}'")

if __name__ == "__main__":
    main()
