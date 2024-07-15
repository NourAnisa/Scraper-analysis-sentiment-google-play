python3.12 -m ensurepip
python3.12 -m pip install setuptools

import subprocess


def install_distutils():
    try:
        import distutils
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])

if __name__ == "__main__":
    install_distutils()
