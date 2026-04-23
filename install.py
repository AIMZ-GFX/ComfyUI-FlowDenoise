import subprocess
import sys


def _install():
    try:
        import memfof  # noqa: F401
        return
    except ImportError:
        pass

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/msu-video-group/memfof.git",
    ])


if __name__ == "__main__":
    _install()
