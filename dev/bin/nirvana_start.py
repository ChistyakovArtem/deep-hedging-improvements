from __future__ import annotations

import json

import torch

from src.nirvana_io import restore_snapshot, write_operation_metadata


def main() -> None:
    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "snapshot": restore_snapshot(),
            },
            indent=2,
        ),
        flush=True,
    )
    write_operation_metadata()


if __name__ == "__main__":
    main()
