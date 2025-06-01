from pathlib import Path

from huggingface_hub import HfApi


def hf_upload_folder(folder_path: Path, save_name: str, token: str, repo_id: str, **kwargs) -> None:
    api = HfApi(token=token)
    api.upload_folder(repo_id=repo_id, folder_path=folder_path, path_in_repo=save_name, **kwargs)


class HFHubManager:
    def __init__(self, token: str, repo_id: str):
        self.token = token
        self.repo_id = repo_id

    def push_to_hub(self, folder_path: Path, save_name: str):
        hf_upload_folder(folder_path=folder_path, save_name=save_name, token=self.token, repo_id=self.repo_id)
