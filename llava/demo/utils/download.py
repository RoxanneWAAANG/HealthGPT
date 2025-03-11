from huggingface_hub import snapshot_download

repo_id = "microsoft/Phi-3-mini-4k-instruct"
local_dir = "/home/jack/Projects/yixin-llm/yixin-llm-data/HealthGPT/Phi3"

# Download the entire repository snapshot to the specified local directory
model_path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
print(f"Model downloaded to: {model_path}")

