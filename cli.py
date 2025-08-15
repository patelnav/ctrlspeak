
import argparse
import sys
from utils.config import get_preferred_model
from models.factory import ModelFactory
from __init__ import __version__

def parse_args_only():
    """Parse command-line arguments without loading heavy libraries."""
    parser = argparse.ArgumentParser(description="ctrlSPEAK - Speech-to-text transcription tool")
    parser.add_argument("--model", type=str, 
                        default=get_preferred_model(),
                        help="Speech recognition model to use (default: %(default)s)")
    parser.add_argument("--source-lang", type=str,
                        default="en",
                        help="Source language for transcription (default: %(default)s)")
    parser.add_argument("--target-lang", type=str,
                        default="en",
                        help="Target language for translation (default: %(default)s)")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with verbose logging")
    parser.add_argument("--check-only", action="store_true",
                        help="Check model cache and configuration, then exit.")
    parser.add_argument("--list-models", action="store_true",
                        help="List all supported models and exit.")
    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {__version__}",
                        help="Show program's version number and exit.")
    
    # Check if -h or --help is present
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit(0)
        
    return parser.parse_args()
