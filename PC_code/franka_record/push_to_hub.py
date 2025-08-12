from huggingface_hub import create_repo, upload_folder

#create_repo(repo_id="ethanCSL/0804_wipe_fix", repo_type="dataset")

upload_folder(
    repo_id="ethanCSL/0804_wipe_fix",
    folder_path="/home/csl/.cache/huggingface/lerobot/ethanCSL/0804_wipe_fix",
    repo_type="dataset",
)
