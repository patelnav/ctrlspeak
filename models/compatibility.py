"""
Compatibility checker for speech-to-text models.
Validates that installed dependencies work with selected models.
"""
import logging
import sys
import platform

logger = logging.getLogger("model_compatibility")


class CompatibilityChecker:
    """Check compatibility of models with installed dependencies."""

    @staticmethod
    def check_nemo_version():
        """
        Check NeMo version and return compatibility info.
        
        Returns:
            dict: {
                'version': str,
                'major': int,
                'minor': int,
                'patch': int,
                'available': bool,
                'error': str or None
            }
        """
        try:
            import nemo
            version_str = nemo.__version__
            
            # Parse version
            parts = version_str.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            
            return {
                'version': version_str,
                'major': major,
                'minor': minor,
                'patch': patch,
                'available': True,
                'error': None
            }
        except ImportError:
            return {
                'version': None,
                'major': None,
                'minor': None,
                'patch': None,
                'available': False,
                'error': 'NeMo not installed'
            }
        except Exception as e:
            return {
                'version': None,
                'major': None,
                'minor': None,
                'patch': None,
                'available': False,
                'error': str(e)
            }

    @staticmethod
    def check_canary_parakeet_compatibility():
        """
        Check if Canary/Parakeet NeMo models will work.
        
        Returns:
            dict: {
                'compatible': bool,
                'nemo_version': str,
                'notes': str,
                'issues': [str]
            }
        """
        nemo_info = CompatibilityChecker.check_nemo_version()
        
        if not nemo_info['available']:
            return {
                'compatible': False,
                'nemo_version': None,
                'notes': 'NeMo toolkit not installed',
                'issues': [nemo_info['error']]
            }
        
        issues = []
        version = nemo_info['version']
        major, minor = nemo_info['major'], nemo_info['minor']
        
        # Known compatibility info:
        # NeMo 1.23.0-1.24.x: Working (returns Hypothesis objects)
        # NeMo 1.25.0-1.x: Breaking change - returns strings instead of Hypothesis objects
        # NeMo 2.0.0+: Returns strings (workaround available)

        if major == 2:
            # NeMo 2.x has changed API - transcribe() returns strings directly
            # This is handled by updated code that accepts both types
            return {
                'compatible': True,
                'nemo_version': version,
                'notes': 'NeMo 2.x uses new API (returns strings). Workaround enabled.',
                'issues': []
            }
        elif major == 1:
            if minor >= 25:
                # 1.25+ returns strings
                issues.append(
                    f"NeMo {version} returns strings instead of Hypothesis objects. "
                    "Workaround enabled to handle both types."
                )
                return {
                    'compatible': True,
                    'nemo_version': version,
                    'notes': 'NeMo 1.25+ API change handled with workaround',
                    'issues': issues
                }
            elif minor >= 23:
                return {
                    'compatible': True,
                    'nemo_version': version,
                    'notes': 'NeMo 1.23-1.24 native compatibility',
                    'issues': []
                }
            else:
                issues.append(f"NeMo {version} is too old. Requires >=1.23.0")
                return {
                    'compatible': False,
                    'nemo_version': version,
                    'notes': 'NeMo version too old',
                    'issues': issues
                }
        else:
            issues.append(f"NeMo {version} (major version {major}) is untested. Compatibility unknown.")
            return {
                'compatible': None,  # Unknown
                'nemo_version': version,
                'notes': 'Version not explicitly tested',
                'issues': issues
            }

    @staticmethod
    def check_mlx_compatibility():
        """
        Check if MLX models will work.
        
        Returns:
            dict: {
                'compatible': bool,
                'notes': str,
                'issues': [str]
            }
        """
        issues = []
        
        if sys.platform != "darwin":
            issues.append("MLX only works on macOS")
            return {
                'compatible': False,
                'notes': 'Not on macOS',
                'issues': issues
            }
        
        if platform.machine() != "arm64":
            issues.append(f"MLX only works on Apple Silicon (arm64), you have {platform.machine()}")
            return {
                'compatible': False,
                'notes': 'Not on Apple Silicon',
                'issues': issues
            }
        
        try:
            import mlx
            import parakeet_mlx
            return {
                'compatible': True,
                'notes': 'MLX available on Apple Silicon',
                'issues': []
            }
        except ImportError as e:
            issues.append(f"MLX dependencies not installed: {e}")
            return {
                'compatible': False,
                'notes': 'MLX not installed',
                'issues': issues
            }

    @staticmethod
    def check_whisper_compatibility():
        """
        Check if Whisper models will work.
        
        Returns:
            dict: {
                'compatible': bool,
                'notes': str,
                'issues': [str]
            }
        """
        issues = []
        
        try:
            import transformers
            import openai_whisper
            return {
                'compatible': True,
                'notes': 'Whisper dependencies installed',
                'issues': []
            }
        except ImportError as e:
            issues.append(f"Whisper dependencies not installed: {e}")
            return {
                'compatible': False,
                'notes': 'Whisper not installed',
                'issues': issues
            }

    @staticmethod
    def diagnose_all():
        """
        Run all compatibility checks and return comprehensive report.
        
        Returns:
            dict: Comprehensive compatibility report
        """
        return {
            'canary_parakeet_nemo': CompatibilityChecker.check_canary_parakeet_compatibility(),
            'mlx': CompatibilityChecker.check_mlx_compatibility(),
            'whisper': CompatibilityChecker.check_whisper_compatibility(),
        }

    @staticmethod
    def print_report():
        """Print a human-readable compatibility report."""
        report = CompatibilityChecker.diagnose_all()
        
        print("\n" + "="*60)
        print("MODEL COMPATIBILITY REPORT")
        print("="*60)
        
        # NeMo models
        nemo = report['canary_parakeet_nemo']
        print(f"\n🔹 NVIDIA Canary/Parakeet (NeMo)")
        print(f"  Status: {'✓ Compatible' if nemo['compatible'] else '✗ Not Compatible' if nemo['compatible'] is False else '⚠ Unknown'}")
        print(f"  Version: {nemo['nemo_version'] or 'Not installed'}")
        print(f"  Notes: {nemo['notes']}")
        if nemo['issues']:
            for issue in nemo['issues']:
                print(f"  Issue: {issue}")
        
        # MLX
        mlx = report['mlx']
        print(f"\n🔹 MLX (Apple Silicon)")
        print(f"  Status: {'✓ Compatible' if mlx['compatible'] else '✗ Not Compatible'}")
        print(f"  Notes: {mlx['notes']}")
        if mlx['issues']:
            for issue in mlx['issues']:
                print(f"  Issue: {issue}")
        
        # Whisper
        whisper = report['whisper']
        print(f"\n🔹 OpenAI Whisper")
        print(f"  Status: {'✓ Compatible' if whisper['compatible'] else '✗ Not Compatible'}")
        print(f"  Notes: {whisper['notes']}")
        if whisper['issues']:
            for issue in whisper['issues']:
                print(f"  Issue: {issue}")
        
        print("\n" + "="*60 + "\n")
