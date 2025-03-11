class Ctrlspeak < Formula
  desc "Minimal speech-to-text utility for macOS"
  homepage "https://github.com/patelnav/ctrlspeak"
  url "https://github.com/patelnav/ctrlspeak/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "4789b0ea9514803bf3200631bf10c2c85f895effbb2ff82cc05e00399fcba071"
  license "MIT"

  depends_on "python@3.11"  # Using Python 3.11 as it's more stable in Homebrew
  depends_on "uv" => :optional  # Optional dependency for faster package installation

  def install
    # Set up virtualenv
    venv = libexec/"venv"
    system "python3.11", "-m", "venv", venv

    # Determine whether to use uv or pip for package installation
    if build.with? "uv"
      # Use UV for faster package installation
      system "uv", "pip", "install", "-r", "requirements.txt", "--prefix", venv
    else
      # Use standard pip
      system venv/"bin/pip", "install", "--upgrade", "pip"
      system venv/"bin/pip", "install", "-r", "requirements.txt"
    end

    # Copy the main script and necessary files
    libexec.install "ctrlspeak.py"
    libexec.install "utils"
    libexec.install "models"
    libexec.install "on.wav"
    libexec.install "off.wav"
    
    # Create a wrapper script that sets up the Python path correctly
    # and also sets the DYLD_LIBRARY_PATH to find the torch and torchaudio libraries
    (bin/"ctrlspeak").write <<~EOS
      #!/bin/bash
      source "#{venv}/bin/activate"
      
      # Set the Python path to include the libexec directory
      export PYTHONPATH="#{libexec}:$PYTHONPATH"
      
      # Set the dynamic library path to find the torch and torchaudio libraries
      TORCH_LIB_PATH="#{venv}/lib/python3.11/site-packages/torch/lib"
      TORCHAUDIO_LIB_PATH="#{venv}/lib/python3.11/site-packages/torchaudio/lib"
      
      # Add both library paths to DYLD_LIBRARY_PATH
      export DYLD_LIBRARY_PATH="$TORCH_LIB_PATH:$TORCHAUDIO_LIB_PATH:$DYLD_LIBRARY_PATH"
      
      # Run the script
      python "#{libexec}/ctrlspeak.py" "$@"
    EOS

    # Make the wrapper executable
    chmod 0755, bin/"ctrlspeak"
  end

  def caveats
    <<~EOS
      To use ctrlSPEAK, grant microphone and accessibility permissions:
      - System Preferences > Security & Privacy > Privacy
      - Add your terminal app (e.g., Terminal.app) to Microphone and Accessibility

      Run `ctrlspeak` in your terminal to start.

      Note: This formula is designed to work with Python 3.11.
      Future versions of Python (3.13+) may require additional dependencies.
    EOS
  end

  test do
    # Add a basic test to check if the script exists and is executable
    assert_predicate bin/"ctrlspeak", :exist?
    assert_predicate bin/"ctrlspeak", :executable?
  end
end 