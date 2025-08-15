
import sys
import subprocess
from rich.panel import Panel
import state
from utils import permission_manager

def check_permissions():
    """Check and request necessary permissions"""
    try:
        state.console.print("\n[bold]Step 1 of 2: Checking microphone access...[/bold]")
        if not permission_manager.check_microphone_permissions(verbose=True, console=state.console):
            state.console.print(Panel.fit(
                "[bold red]Microphone access required[/bold red]\n\n"\
                "ctrlspeak needs microphone access to record your speech.\n"\
                "Without this permission, the app cannot transcribe audio.\n\n"\
                "[yellow]Opening System Settings → Privacy & Security → Microphone...[/yellow]\n"\
                "Please add and enable this application in the list.",
                title="Permission Required",
                border_style="red"
            ))
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"])
            state.console.print("\nPlease restart the application after granting permission.")
            sys.exit(1)
        else:
            state.console.print("[bold green]✓ Microphone access is granted.[/bold green]")
    except Exception as e:
        state.console.print(f"[bold red]Error accessing microphone: {e}[/bold red]")
        sys.exit(1)

    state.console.print("\n[bold]Step 2 of 2: Checking keyboard monitoring permissions...[/bold]")
    if not permission_manager.check_keyboard_permissions(verbose=True, console=state.console):
        state.console.print("\nPlease restart the application after granting permission.")
        sys.exit(1)
    
    state.console.print("\n[bold green]All required permissions are granted! Starting ctrlSPEAK...[/bold green]")
    
    return True
