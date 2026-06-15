#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run native benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mllm_kvcompress.benchmarks.native_runner import main


if __name__ == "__main__":
    main()
