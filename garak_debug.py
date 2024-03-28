#!/usr/bin/env python3
# requires install of debugpy not in requirements

import sys
import time
import garak
from garak import cli
import debugpy

import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

def main():
    debugpy.listen(("localhost", 5678))
    time.sleep(5)
    cli.main(sys.argv[1:])


if __name__ == "__main__":
    main()
