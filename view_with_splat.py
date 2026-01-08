#!/usr/bin/env python3
"""
Helper script to view .ply files using the splat viewer (https://antimatter15.com/splat/)

This script can:
1. Convert .ply files to .splat format (if splat repository is available)
2. Open .ply files in the web viewer
3. Provide instructions for manual viewing

Usage:
    python view_with_splat.py <path_to_ply_file>
    python view_with_splat.py <path_to_ply_file> --convert
    python view_with_splat.py <path_to_ply_file> --open
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path


def check_splat_repo():
    """Check if splat repository is available locally."""
    splat_path = Path("./splat")
    if splat_path.exists() and (splat_path / "convert.py").exists():
        return splat_path / "convert.py"
    return None


def convert_ply_to_splat(ply_path, output_path=None):
    """
    Convert .ply file to .splat format using the splat library's convert.py script.
    
    Args:
        ply_path: Path to input .ply file
        output_path: Path to output .splat file (optional, defaults to same name as .ply)
    
    Returns:
        Path to converted .splat file or None if conversion failed
    """
    convert_script = check_splat_repo()
    
    if convert_script is None:
        print("Splat repository not found locally.")
        print("To convert .ply to .splat format, you can:")
        print("1. Clone the splat repository: git clone https://github.com/antimatter15/splat.git")
        print("2. Or use the web viewer directly by dragging your .ply file onto https://antimatter15.com/splat/")
        return None
    
    # Check if plyfile is available in the current Python environment
    try:
        import plyfile
    except ImportError:
        print("Error: 'plyfile' module is not installed in the current Python environment.")
        print(f"Current Python: {sys.executable}")
        print("Please install it with: pip install plyfile")
        print("Or activate your conda environment (e.g., 'conda activate diffusiongs') that has plyfile installed.")
        return None
    
    if output_path is None:
        output_path = str(Path(ply_path).with_suffix('.splat'))
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert paths to absolute strings
        ply_path_str = str(Path(ply_path).absolute())
        output_path_str = str(output_path_obj.absolute())
        
        print(f"Converting {ply_path_str} to {output_path_str}...")
        result = subprocess.run(
            [sys.executable, str(convert_script), ply_path_str, "--output", output_path_str],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Successfully converted to {output_path_str}")
        return output_path_str
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None


def open_in_web_viewer(ply_path, convert_first=False):
    """
    Open .ply file in the splat web viewer.
    
    Args:
        ply_path: Path to .ply file
        convert_first: If True, convert to .splat first and open the .splat file
    """
    ply_path = Path(ply_path)
    
    if not ply_path.exists():
        print(f"Error: File not found: {ply_path}")
        return
    
    if convert_first:
        splat_path = convert_ply_to_splat(ply_path)
        if splat_path:
            # If file is hosted locally, you'd need to serve it via HTTP
            # For now, just provide instructions
            print(f"\nConverted file saved to: {splat_path}")
            print("To view it in the splat viewer:")
            print(f"1. Host the file on a web server with CORS enabled")
            print(f"2. Visit: https://antimatter15.com/splat/?url=<your_url_to_file.splat>")
        else:
            print("\nConversion failed. You can still use the drag-and-drop method.")
    else:
        # Provide instructions for drag-and-drop
        print(f"\nTo view {ply_path.name} in the splat viewer:")
        print("1. Open https://antimatter15.com/splat/ in your browser")
        print(f"2. Drag and drop {ply_path.absolute()} onto the page")
        print("3. The file will be automatically converted and displayed")
        
        # Optionally try to open the browser
        try:
            response = input("\nWould you like to open the splat viewer in your browser? (y/n): ")
            if response.lower() == 'y':
                webbrowser.open('https://antimatter15.com/splat/')
        except KeyboardInterrupt:
            print("\nCancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="View .ply files using the splat viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a .ply file using drag-and-drop method
  python view_with_splat.py output.ply
  
  # Convert .ply to .splat format (requires splat repository)
  python view_with_splat.py output.ply --convert
  
  # Open web viewer in browser
  python view_with_splat.py output.ply --open
        """
    )
    parser.add_argument(
        "ply_file",
        type=str,
        help="Path to .ply file to view"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert .ply to .splat format (requires splat repository)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the splat web viewer in browser"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .splat file (only used with --convert)"
    )
    
    args = parser.parse_args()
    
    ply_path = Path(args.ply_file)
    
    if not ply_path.exists():
        print(f"Error: File not found: {ply_path}")
        sys.exit(1)
    
    if not ply_path.suffix.lower() == '.ply':
        print(f"Warning: File does not have .ply extension: {ply_path}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    if args.convert:
        output_path = convert_ply_to_splat(ply_path, args.output)
        if output_path:
            print(f"\n✓ Conversion complete: {output_path}")
        else:
            print("\n✗ Conversion failed. See instructions above.")
    else:
        open_in_web_viewer(ply_path, convert_first=False)
        if args.open:
            try:
                webbrowser.open('https://antimatter15.com/splat/')
            except Exception as e:
                print(f"Could not open browser: {e}")


if __name__ == "__main__":
    main()

