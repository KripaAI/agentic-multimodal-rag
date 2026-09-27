"""Phase 0: confirm the Kaggle GPU is reachable (runs on Kaggle, not locally)."""
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except Exception as e:
    print("torch check failed:", e)
