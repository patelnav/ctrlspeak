class Ctrlspeak < Formula
  desc "Minimal speech-to-text utility for macOS"
  homepage "https://github.com/navpatel/stt"
  url "https://github.com/navpatel/stt/archive/v1.0.0.tar.gz"
  sha256 "your-sha256-checksum-here"  # You'll need to replace this with the actual checksum

  depends_on "python@3.11"

  def install
    # Set up virtualenv
    venv = libexec/"venv"
    system "python3.11", "-m", "venv", venv

    # Install dependencies
    system venv/"bin/pip", "install", "-r", "requirements.txt"

    # Copy the main script and necessary files
    libexec.install "ctrlspeak.py"
    libexec.install "utils"
    libexec.install "on.wav"
    libexec.install "off.wav"
    
    # Create models directory
    (libexec/"models").mkpath

    # Create a wrapper script
    (bin/"ctrlspeak").write <<~EOS
      #!/bin/bash
      source "#{venv}/bin/activate"
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
    EOS
  end

  test do
    # Add a basic test to check if the script exists and is executable
    assert_predicate bin/"ctrlspeak", :exist?
    assert_predicate bin/"ctrlspeak", :executable?
  end
end 