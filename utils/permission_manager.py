"""
Permission Manager for ctrlSPEAK

Handles checking and requesting various permissions required by the application,
with a focus on accurate detection of accessibility permissions.
"""

import sys
import os
import time
import subprocess
import threading
import re
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel

# Default console
_console = Console()

# Permissions state
_permissions = {
    "keyboard": {
        "granted": False,
        "working": False,
        "errors": []
    },
    "microphone": {
        "granted": False,
        "working": False,
        "errors": []
    }
}

# Cache parent app name
_parent_app = None

# Import Apple's accessibility API for direct permission checking
_ax_api_available = False
try:
    import objc
    from Foundation import NSBundle
    from ApplicationServices import AXIsProcessTrusted
    
    # If we've made it this far, the API is available
    _ax_api_available = True
except (ImportError, AttributeError, ValueError):
    # If we can't import the required modules, we'll fall back to behavior testing
    pass

# Try to import the microphone permission checking function
_mic_check_available = False
try:
    # More robust import that won't be affected by circular imports
    import sounddevice as sd
    
    def check_mic_permissions():
        """Check microphone permissions directly"""
        try:
            with sd.InputStream(channels=1, callback=lambda *args: None):
                pass
            return True
        except sd.PortAudioError:
            return False
    
    _mic_check_available = True
except ImportError:
    pass

# Also try to import from utils.audio as fallback
try:
    from utils.audio import check_microphone_permissions as audio_check_mic
    _audio_mic_check_available = True
except ImportError:
    _audio_mic_check_available = False

def detect_parent_app():
    """
    Detect which application is running this script.
    
    Returns:
        str: Name of the parent application or terminal
    """
    global _parent_app
    
    # Return cached result if available
    if _parent_app:
        return _parent_app
    
    # First check if TERM_PROGRAM environment variable is set
    term_program = os.environ.get('TERM_PROGRAM')
    if term_program:
        _parent_app = term_program  # This will be "Apple_Terminal", "iTerm.app", etc.
        return _parent_app
    
    # Try to get the parent process info using ps command
    try:
        ppid = os.getppid()
        result = subprocess.run(
            ['ps', '-p', str(ppid), '-o', 'comm='],
            capture_output=True, 
            text=True,
            check=True
        )
        parent_process = result.stdout.strip()
        
        # Extract just the program name without path
        match = re.search(r'[^/]+$', parent_process)
        if match:
            _parent_app = match.group(0)
            return _parent_app
        
        _parent_app = parent_process
        return _parent_app
    except Exception:
        # If we can't detect it, return a generic description
        _parent_app = "terminal application"
        return _parent_app

def check_all_permissions(verbose=True, console=None):
    """
    Check all required permissions.
    
    Args:
        verbose (bool): Whether to print detailed output
        console (Console, optional): Rich console instance to use for output
        
    Returns:
        bool: True if all permissions are granted, False otherwise
    """
    console = console or _console
    
    if verbose:
        console.print("\n[bold]Checking all required permissions...[/bold]")
    
    # Check keyboard permissions
    keyboard_ok = check_keyboard_permissions(verbose=verbose, console=console)
    
    # Check microphone permissions
    mic_ok = check_microphone_permissions(verbose=verbose, console=console)
    
    if verbose:
        if keyboard_ok and mic_ok:
            console.print("[bold green]✓ All required permissions are granted.[/bold green]")
        else:
            console.print("[bold red]✗ Some permissions are missing.[/bold red]")
    
    # Return True only if all permissions are granted
    return keyboard_ok and mic_ok

def check_keyboard_permissions(verbose=True, console=None):
    """
    Check if keyboard monitoring permissions are granted.
    
    Args:
        verbose (bool): Whether to print detailed output
        console (Console, optional): Rich console instance to use for output
        
    Returns:
        bool: True if permissions are granted, False otherwise
    """
    console = console or _console
    parent_app = detect_parent_app()
    
    if verbose:
        console.print("[bold]Checking keyboard monitoring permissions...[/bold]")
        console.print(f"[dim]Running from: {parent_app}[/dim]")
    
    # Reset keyboard permission state
    _permissions["keyboard"] = {
        "granted": False,
        "working": False,
        "errors": []
    }
    
    # Use direct Apple API first if available (most reliable method)
    if _ax_api_available:
        if verbose:
            console.print("Using macOS Accessibility API directly...")
        
        permission_granted = AXIsProcessTrusted()
        
        if permission_granted:
            if verbose:
                console.print("[green]✓[/green] Accessibility permissions granted (via AXIsProcessTrusted)")
            _permissions["keyboard"]["granted"] = True
            _permissions["keyboard"]["working"] = True
            return True
        else:
            if verbose:
                console.print("[bold red]✗[/bold red] Accessibility permissions not granted (via AXIsProcessTrusted)")
                request_keyboard_permissions(console=console)
            return False
    else:
        if verbose:
            console.print("[yellow]⚠[/yellow] macOS Accessibility API not available, falling back to behavioral tests")
    
    # Fall back to behavior checks if API not available
    if verbose:
        console.print("Creating keyboard listener...")
    
    try:
        test_listener = keyboard.Listener(on_press=lambda k: None)
        test_listener.start()
        time.sleep(0.5)  # Give it a moment to fail if it's going to
        
        listener_works = test_listener.is_alive()
        if listener_works:
            if verbose:
                console.print("[green]✓[/green] Keyboard listener created successfully")
        else:
            if verbose:
                console.print("[bold red]✗[/bold red] Keyboard listener creation failed")
            
            # If we can't even create a listener, accessibility permissions are definitely not granted
            error_msg = "Cannot create keyboard listener. Accessibility permissions are not granted."
            _permissions["keyboard"]["errors"].append(error_msg)
            if verbose:
                console.print(f"[bold red]✗[/bold red] {error_msg}")
                request_keyboard_permissions(console=console)
            
            test_listener.stop()
            return False
        
        test_listener.stop()
    except Exception as e:
        # If this fails, permissions are definitely not granted
        error_msg = f"Error creating keyboard listener: {str(e)}"
        _permissions["keyboard"]["errors"].append(error_msg)
        if verbose:
            console.print(f"[bold red]✗[/bold red] {error_msg}")
            request_keyboard_permissions(console=console)
        return False
    
    # Test 2: Try to simulate a key event (automated test)
    if verbose:
        console.print("Testing keyboard event detection...")
    
    controller_works = False
    
    try:
        # Create a test event that will be set when a key is detected
        key_detected = threading.Event()
        def on_key_test(key):
            key_detected.set()
        
        # Start a listener
        test_listener = keyboard.Listener(on_press=on_key_test)
        test_listener.start()
        
        try:
            # Try to simulate a keyboard event
            controller = keyboard.Controller()
            controller.press(keyboard.Key.shift)
            controller.release(keyboard.Key.shift)
            
            # Wait up to 1 second for the event to be detected
            if key_detected.wait(timeout=1.0):
                controller_works = True
                if verbose:
                    console.print("[green]✓[/green] Keyboard events can be detected")
            else:
                if verbose:
                    console.print("[yellow]⚠[/yellow] Failed to detect simulated keyboard events")
        except Exception as e:
            if verbose:
                console.print(f"[yellow]⚠[/yellow] Keyboard controller error: {str(e)}")
        finally:
            if test_listener.is_alive():
                test_listener.stop()
    
    except Exception as e:
        if verbose:
            console.print(f"[yellow]⚠[/yellow] Keyboard test error: {str(e)}")
    
    # Make the final determination based on behavioral tests
    if controller_works:
        # If controller works, we probably have permissions
        _permissions["keyboard"]["granted"] = True
        _permissions["keyboard"]["working"] = True
        if verbose:
            console.print("[bold green]✓ Keyboard monitoring permissions appear to be working.[/bold green]")
        return True
    elif listener_works:
        # If only listener works but simulated events don't, we have uncertain status
        _permissions["keyboard"]["granted"] = True  # Being optimistic
        _permissions["keyboard"]["working"] = False
        if verbose:
            console.print("[bold yellow]⚠ Keyboard permissions partially working.[/bold yellow]")
            console.print("The application may have limited keyboard monitoring capabilities.")
        return True
    else:
        # Nothing works
        _permissions["keyboard"]["granted"] = False
        _permissions["keyboard"]["working"] = False
        if verbose:
            console.print("[bold red]✗ Keyboard monitoring permissions are not granted.[/bold red]")
            request_keyboard_permissions(console=console)
        return False

def request_keyboard_permissions(console=None):
    """Show permission request panel and open System Settings for keyboard accessibility"""
    console = console or _console
    parent_app = detect_parent_app()
    
    console.print(Panel.fit(
        f"[bold red]Keyboard monitoring permissions required[/bold red]\n\n"
        f"ctrlSPEAK needs Accessibility permissions to detect keyboard shortcuts.\n"
        f"Without this permission, the app cannot detect when you triple-tap Ctrl.\n\n"
        f"[yellow]IMPORTANT:[/yellow] When running from {parent_app}, you need to grant\n"
        f"accessibility permissions to [bold]{parent_app}[/bold], not Python or ctrlSPEAK.\n\n"
        f"[yellow]Opening System Settings → Privacy & Security → Accessibility...[/yellow]\n"
        f"Please find and enable {parent_app} in the list.",
        title="Permission Required",
        border_style="red"
    ))
    
    # Open System Settings based on platform
    if sys.platform == "darwin":  # macOS
        subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])
    elif sys.platform == "win32":  # Windows
        subprocess.run(["start", "ms-settings:privacy-accessibilityinapp"], shell=True)
    
    console.print("\n[bold]After granting permission:[/bold]")
    console.print(f"1. Make sure [bold]{parent_app}[/bold] is checked in the list")
    console.print("2. You may need to restart the application")

def request_microphone_permissions(console=None):
    """Show permission request panel and open System Settings for microphone access"""
    console = console or _console
    parent_app = detect_parent_app()
    
    console.print(Panel.fit(
        f"[bold red]Microphone access required[/bold red]\n\n"
        f"ctrlSPEAK needs microphone access to record your speech.\n"
        f"Without this permission, the app cannot transcribe audio.\n\n"
        f"[yellow]Opening System Settings → Privacy & Security → Microphone...[/yellow]\n"
        f"Please add and enable this application in the list.",
        title="Permission Required",
        border_style="red"
    ))
    
    # Open System Settings based on platform
    if sys.platform == "darwin":  # macOS
        subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"])
    elif sys.platform == "win32":  # Windows
        subprocess.run(["start", "ms-settings:privacy-microphone"], shell=True)
    
    console.print("\n[bold]After granting permission:[/bold]")
    console.print(f"1. Make sure the microphone access is enabled for this application")
    console.print("2. You may need to restart the application")

def check_microphone_permissions(verbose=True, console=None):
    """
    Check if microphone permissions are granted.
    
    Args:
        verbose (bool): Whether to print detailed output
        console (Console, optional): Rich console instance to use for output
        
    Returns:
        bool: True if permissions are granted, False otherwise
    """
    console = console or _console
    
    if verbose:
        console.print("[bold]Checking microphone permissions...[/bold]")
    
    # Reset microphone permission state
    _permissions["microphone"] = {
        "granted": False,
        "working": False,
        "errors": []
    }
    
    # First try our direct sounddevice check
    if _mic_check_available:
        try:
            mic_permitted = check_mic_permissions()
            if mic_permitted:
                if verbose:
                    console.print("[green]✓[/green] Microphone permissions granted")
                _permissions["microphone"]["granted"] = True
                _permissions["microphone"]["working"] = True
                return True
            else:
                if verbose:
                    console.print("[bold red]✗[/bold red] Microphone permissions not granted")
                    request_microphone_permissions(console=console)
                return False
        except Exception as e:
            error_msg = f"Error checking microphone permissions directly: {str(e)}"
            _permissions["microphone"]["errors"].append(error_msg)
            if verbose:
                console.print(f"[bold red]✗[/bold red] {error_msg}")
            
            # If the direct check fails, try the fallback
    
    # Fallback to the existing audio module function if available
    if _audio_mic_check_available:
        try:
            mic_permitted = audio_check_mic()
            if mic_permitted:
                if verbose:
                    console.print("[green]✓[/green] Microphone permissions granted")
                _permissions["microphone"]["granted"] = True
                _permissions["microphone"]["working"] = True
                return True
            else:
                if verbose:
                    console.print("[bold red]✗[/bold red] Microphone permissions not granted")
                    request_microphone_permissions(console=console)
                return False
        except Exception as e:
            error_msg = f"Error checking microphone permissions via audio module: {str(e)}"
            _permissions["microphone"]["errors"].append(error_msg)
            if verbose:
                console.print(f"[bold red]✗[/bold red] {error_msg}")
            return False
    else:
        # Placeholder basic check if the audio module is not available
        if verbose:
            console.print("[yellow]⚠[/yellow] Audio module not available, microphone status unknown")
            console.print("[green]✓[/green] Assuming microphone permissions for now")
        
        _permissions["microphone"]["granted"] = True
        _permissions["microphone"]["working"] = True
        return True

def get_permissions_status():
    """
    Get the current permissions status.
    
    Returns:
        dict: The current permissions status
    """
    return _permissions.copy() 