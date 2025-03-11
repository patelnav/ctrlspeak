#!/usr/bin/env python3
"""
Permission Testing Utility for ctrlSPEAK

A comprehensive testing tool that checks all required permissions
and verifies they're working correctly. This is useful for debugging
permission issues during development or helping users troubleshoot.
"""

import sys
import os
import time
import platform
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

# Import our permission manager functions
from utils import permission_manager

class PermissionTester:
    def __init__(self):
        self.console = Console()
        
    def run_all_tests(self):
        """Run all permission tests"""
        self.console.print(Panel.fit(
            "[bold]ctrlSPEAK Permission Tester[/bold]\n\n"
            "This utility will check if all required permissions are correctly granted "
            "and working properly.",
            title="Permission Test",
            border_style="blue"
        ))
        
        self.console.print("\n[bold]System Information:[/bold]")
        self.console.print(f"OS: {platform.system()} {platform.release()}")
        self.console.print(f"Python: {sys.version.split()[0]}")
        self.console.print(f"Running from: {sys.executable}")
        self.console.print(f"Parent Application: [bold]{permission_manager.detect_parent_app()}[/bold]")
        self.console.print(f"Script location: {__file__}")
        
        # Check all permissions using our manager
        self.console.print("\n[bold cyan]Testing All Permissions[/bold cyan]")
        all_permissions_ok = permission_manager.check_all_permissions(verbose=True, console=self.console)
        
        # Display summary
        self.display_summary()
        
        return all_permissions_ok
    
    def test_specific_permission(self, permission_type):
        """Test a specific permission type"""
        if permission_type == "keyboard":
            self.console.print("\n[bold cyan]Testing Keyboard Permissions[/bold cyan]")
            return permission_manager.check_keyboard_permissions(verbose=True, console=self.console)
        elif permission_type == "microphone":
            self.console.print("\n[bold cyan]Testing Microphone Permissions[/bold cyan]")
            return permission_manager.check_microphone_permissions(verbose=True, console=self.console)
        else:
            self.console.print(f"[bold red]✗[/bold red] Unknown permission type: {permission_type}")
            return False

    def display_summary(self):
        """Display a summary of all test results"""
        self.console.print("\n[bold]Test Results Summary:[/bold]")
        
        # Get permission states
        permissions_status = permission_manager.get_permissions_status()
        
        keyboard_granted = permissions_status["keyboard"]["granted"]
        keyboard_working = permissions_status["keyboard"]["working"]
        
        microphone_granted = permissions_status["microphone"]["granted"]
        microphone_working = permissions_status["microphone"]["working"]
        
        # Microphone summary
        mic_status = "[green]PASSED[/green]" if (
            microphone_granted and microphone_working
        ) else "[red]FAILED[/red]"
        
        self.console.print(f"Microphone: {mic_status}")
        self.console.print(f"  - Permission Granted: {'[green]Yes[/green]' if microphone_granted else '[red]No[/red]'}")
        self.console.print(f"  - Functionality Working: {'[green]Yes[/green]' if microphone_working else '[red]No[/red]'}")
        
        if permissions_status["microphone"]["errors"]:
            self.console.print("  - Errors:")
            for error in permissions_status["microphone"]["errors"]:
                self.console.print(f"    - {error}")
        
        # Keyboard summary
        kb_status = "[green]PASSED[/green]" if (
            keyboard_granted and keyboard_working
        ) else "[red]FAILED[/red]"
        
        self.console.print(f"\nKeyboard Monitoring: {kb_status}")
        self.console.print(f"  - Permission Granted: {'[green]Yes[/green]' if keyboard_granted else '[red]No[/red]'}")
        self.console.print(f"  - Functionality Working: {'[green]Yes[/green]' if keyboard_working else '[red]No[/red]'}")
        
        if permissions_status["keyboard"]["errors"]:
            self.console.print("  - Errors:")
            for error in permissions_status["keyboard"]["errors"]:
                self.console.print(f"    - {error}")
        
        # Overall result
        overall_result = (
            (keyboard_granted and keyboard_working) and 
            (microphone_granted and microphone_working)
        )
        
        self.console.print(f"\n[bold]{'[green]ALL TESTS PASSED[/green]' if overall_result else '[red]TESTS FAILED[/red]'}[/bold]")
        
        if not overall_result:
            self.console.print("\n[yellow]Troubleshooting Tips:[/yellow]")
            parent_app = permission_manager.detect_parent_app()
            
            if not keyboard_granted:
                self.console.print(f"- Make sure [bold]{parent_app}[/bold] has accessibility permissions in System Settings")
                
                if sys.platform == "darwin":  # macOS specific
                    self.console.print("- For accessibility permissions on macOS:")
                    self.console.print("  1. Go to System Settings → Privacy & Security → Accessibility")
                    self.console.print(f"  2. Make sure [bold]{parent_app}[/bold] is CHECKED (not Python or ctrlSPEAK)")
                    self.console.print("  3. If already checked, try removing and re-adding the permission")
                    self.console.print("  4. Log out and log back in, or restart your computer")
            
            if not microphone_granted:
                self.console.print("- Make sure the application has microphone permissions in System Settings")
                
                if sys.platform == "darwin":  # macOS specific
                    self.console.print("- For microphone permissions on macOS:")
                    self.console.print("  1. Go to System Settings → Privacy & Security → Microphone")
                    self.console.print("  2. Make sure the application is in the list and CHECKED")

def main():
    """Run the permission tests"""
    tester = PermissionTester()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # If specific permission is specified, just test that one
        permission_type = sys.argv[1].lower()
        success = tester.test_specific_permission(permission_type)
    else:
        # Otherwise test all permissions
        success = tester.run_all_tests()
    
    # Return appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 