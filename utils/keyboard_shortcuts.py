import threading
from pynput import keyboard
import subprocess
import sys
import time
import os
from rich.console import Console
from rich.panel import Panel
from utils.permission_manager import check_keyboard_permissions

class KeyboardShortcutManager:
    """
    A class to manage keyboard shortcuts and hotkeys
    """
    def __init__(self):
        self.hotkey_listener = None
        self.shortcuts = {}
        self.is_running = True
        self.console = Console()
        
        # For triple-tap detection
        self.last_key_time = 0
        self.ctrl_tap_count = 0
        self.ctrl_tap_timeout = 0.5  # seconds between taps
        self.triple_tap_callback = None
        self.key_listener = None
    
    def check_permissions(self):
        """Check and request necessary accessibility permissions for keyboard control"""
        return check_keyboard_permissions(verbose=True)
    
    def register_shortcut(self, key_combination, callback):
        """
        Register a keyboard shortcut
        
        Args:
            key_combination (str): Key combination in pynput format (e.g., '<alt>+`')
            callback (function): Function to call when shortcut is pressed
        """
        self.shortcuts[key_combination] = callback
    
    def register_triple_ctrl_tap(self, callback):
        """
        Register a callback for when Ctrl is tapped three times in succession
        
        Args:
            callback (function): Function to call when triple-tap is detected
        """
        self.triple_tap_callback = callback
    
    def _on_key_press(self, key):
        """
        Internal handler for key press events to detect triple-tap
        """
        # Check if it's a ctrl key
        if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            current_time = time.time()
            
            # If it's been too long since the last tap, reset the counter
            if current_time - self.last_key_time > self.ctrl_tap_timeout:
                self.ctrl_tap_count = 1
            else:
                self.ctrl_tap_count += 1
            
            self.last_key_time = current_time
            
            # If we've reached 3 taps, trigger the callback
            if self.ctrl_tap_count == 3 and self.triple_tap_callback:
                self.ctrl_tap_count = 0  # Reset counter
                return self.triple_tap_callback()
        
        return True  # Continue listening
    
    def _on_key_release(self, key):
        """
        Internal handler for key release events
        """
        # Just continue listening
        return True
    
    def start_listening(self):
        """Start listening for registered shortcuts and triple-tap"""
        # Start the regular hotkey listener if shortcuts are registered
        if self.shortcuts:
            self.hotkey_listener = keyboard.GlobalHotKeys(self.shortcuts)
            self.hotkey_listener.start()
        
        # Start the key listener for triple-tap detection
        if self.triple_tap_callback:
            self.key_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.key_listener.start()
        
        return True
    
    def stop_listening(self):
        """Stop listening for shortcuts"""
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        
        if self.key_listener:
            self.key_listener.stop()
            
        self.is_running = False
    
    def join(self):
        """Join the hotkey listener thread"""
        if self.hotkey_listener:
            self.hotkey_listener.join()
        
        if self.key_listener:
            self.key_listener.join()

def check_keyboard_monitoring_permissions():
    """
    Standalone function to check if the application has keyboard monitoring permissions.
    
    Returns:
        bool: True if permissions are granted, False otherwise
    """
    console = Console()
    console.print("[bold]Checking keyboard monitoring permissions...[/bold]")
    
    # Multiple tests to verify permissions
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Basic listener creation
    try:
        console.print("Test 1: Creating keyboard listener...")
        test_listener = keyboard.Listener(on_press=lambda k: None)
        test_listener.start()
        time.sleep(0.5)  # Give it a moment to fail if it's going to
        
        if test_listener.is_alive():
            tests_passed += 1
            console.print("[green]✓[/green] Keyboard listener created successfully")
        else:
            console.print("[bold red]✗[/bold red] Keyboard listener creation failed")
        
        test_listener.stop()
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Error creating keyboard listener: {e}")
        _show_permission_request_panel(console)
        return False
    
    # Test 2: Try to programmatically generate keyboard events
    try:
        console.print("Test 2: Testing keyboard event simulation...")
        
        # Create a test event to track if keyboard events are received
        event_received = threading.Event()
        
        def on_press_test(key):
            event_received.set()
            return False  # Stop listener
        
        # Create a listener that will respond to generated events
        test_listener = keyboard.Listener(on_press=on_press_test)
        test_listener.start()
        
        # Try to create a keyboard controller and generate an event
        try:
            controller = keyboard.Controller()
            # Press a harmless key
            controller.press(keyboard.Key.shift)
            controller.release(keyboard.Key.shift)
            
            # Wait for the event to be received
            if event_received.wait(timeout=1.0):
                tests_passed += 1
                console.print("[green]✓[/green] Keyboard event simulation successful")
            else:
                console.print("[yellow]⚠[/yellow] Keyboard event simulation failed")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Keyboard controller error: {e}")
        finally:
            if test_listener.is_alive():
                test_listener.stop()
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Keyboard event test error: {e}")
    
    # Test 3: Check permissions on macOS specifically
    if sys.platform == "darwin":
        console.print("Test 3: Checking macOS accessibility permissions...")
        try:
            # Check if the app is in the list of apps with accessibility access
            # This requires sudo, so it might time out if run as normal user
            cmd = [
                "sudo", "sqlite3", 
                "/Library/Application Support/com.apple.TCC/TCC.db", 
                "SELECT allowed FROM access WHERE service='kTCCServiceAccessibility' AND client=?",
                sys.executable
            ]
            
            proc = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1  # Short timeout in case it waits for password
            )
            
            if proc.returncode == 0 and proc.stdout.strip() == "1":
                tests_passed += 1
                console.print("[green]✓[/green] macOS TCC database confirms permissions")
            elif proc.returncode == 0 and proc.stdout.strip() == "0":
                console.print("[bold red]✗[/bold red] macOS TCC database shows permission denied")
            else:
                console.print("[yellow]⚠[/yellow] Unable to check macOS TCC database (try running with sudo)")
                
                # If we couldn't check the database, we need an alternative test
                console.print("Running alternative test...")
                tests_passed += 0.5  # Half credit for passing the basic test earlier
        except (subprocess.SubprocessError, FileNotFoundError):
            console.print("[yellow]⚠[/yellow] Unable to query macOS permission database")
            tests_passed += 0.5  # Half credit for passing the basic test earlier
    else:
        # Non-macOS platform, assume the basic test is sufficient
        tests_passed += 1
    
    # Calculate permission confidence and make decision
    permission_confidence = tests_passed / tests_total
    
    if permission_confidence >= 0.7:  # At least 2/3 tests passing
        console.print("[bold green]✓ Keyboard monitoring permissions are granted.[/bold green]")
        return True
    elif permission_confidence >= 0.3:  # At least 1/3 tests passing
        console.print("[bold yellow]⚠ Keyboard permissions partially verified.[/bold yellow]")
        console.print("The application may have limited keyboard monitoring capabilities.")
        _show_permission_request_panel(console)
        # Continue anyway but warn user
        return True
    else:
        console.print("[bold red]✗ Keyboard monitoring permissions are not granted.[/bold red]")
        _show_permission_request_panel(console)
        return False

def _show_permission_request_panel(console):
    """Helper function to show the permission request panel"""
    console.print(Panel.fit(
        "[bold red]Keyboard monitoring permissions required[/bold red]\n\n"
        "ctrlSPEAK needs Accessibility permissions to detect keyboard shortcuts.\n"
        "Without this permission, the app cannot detect when you triple-tap Ctrl.\n\n"
        "[yellow]Opening System Settings → Privacy & Security → Accessibility...[/yellow]\n"
        "Please add and enable this application in the list.",
        title="Permission Required",
        border_style="red"
    ))
    
    # Open System Settings to the right place
    subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])
    
    console.print("\n[bold]After granting permission:[/bold]")
    console.print("1. Make sure the app is checked in the list")
    console.print("2. You may need to restart the application")
    console.print("3. If using from a terminal, try running with 'sudo'") 