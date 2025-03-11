"""
Clipboard operations module for ctrlSPEAK.
"""
import time
import pyperclip

def copy_to_clipboard(text):
    """Copy text to clipboard"""
    pyperclip.copy(text)

def paste_from_clipboard():
    """Simulate Command+V to paste from clipboard"""
    # Create keyboard controller
    from pynput import keyboard
    kb = keyboard.Controller()
    
    # Small delay to ensure clipboard is ready
    time.sleep(0.1)
    
    # Simulate Command+V to paste
    with kb.pressed(keyboard.Key.cmd):
        kb.press('v')
        kb.release('v') 