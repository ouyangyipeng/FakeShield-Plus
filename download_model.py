import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 代理保持原样
os.environ["HTTP_PROXY"] = "http://10.186.75.4:3128"
os.environ["HTTPS_PROXY"] = "http://10.186.75.4:3128"

# ❌ 关键：注释掉这行，禁用 hf_transfer
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "zhipeixu/fakeshield-v1-22b"
local_dir = "./weight/fakeshield-v1-22b"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,      # 启用续传
    max_workers=4,             # 并发数不宜过高，避免代理压力
    # 可选：跳过已经完整下载的文件，可以大幅提速
    ignore_patterns=["*.partial"]   # 忽略未完成的临时文件
)