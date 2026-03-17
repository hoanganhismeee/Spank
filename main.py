"""
main.py — Entry point for Spank.

Usage:
    python main.py              # run the detector
    python main.py --calibrate  # run the calibration wizard first
"""

import sys
from detector import Calibrator, main


def run() -> None:
    if "--calibrate" in sys.argv or "-c" in sys.argv:
        Calibrator().run()
    main()


if __name__ == "__main__":
    run()
