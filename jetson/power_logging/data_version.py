"""Dagshub Client."""

import dagshub


class DagsHubClient:
    """DagsHub Client.

    Attributes:
        owner: User or org that owns the repository.
        name: Name of the repository.
        dagshub_repo: RepoAPI to fetch details of the repository.
    """

    def __init__(self, owner: str, name: str) -> None:
        """Init function for DagsHubClient class.

        DagsHub uses tokens for authentication.
        `DAGSHUB_USER_TOKEN` environment variable can be set with token value.
        If no token is provided, it will trigger a OAuth authenticator
        to fetch the token explicitly.

        More information: https://dagshub.com/docs/client/reference/auth.html#authentication

        Args:
            owner: User or org that owns the repository.
            name: Name of the repository.
        """
        self.owner = owner
        self.name = name
        repo_url = f"{self.owner}/{self.name}"
        self.dagshub_repo = dagshub.common.api.repo.RepoAPI(repo=repo_url)

    def upload_dataset(
        self,
        dataset_dir: str,
        commit_message: str,
        branch_name: str,
        versioning: str = "dvc",
    ) -> None:
        """Upload directory to Dagshub using DVC as data versioning.

        Args:
            dataset_dir: Path to local directory to be uploaded
            commit_message: Corresponding commit message to be added.
            branch_name: Name of branch to use for the dataset.
            versioning: Data versioning approach. Defaults to "dvc".
        """
        repo = dagshub.upload.Repo(owner=self.owner, name=self.name, branch=branch_name)

        # Get last commit to overwrite data if pushed multiple times
        last_commit = None
        try:
            last_commit = self.dagshub_repo.last_commit(branch=branch_name).id
        except:
            print("This looks like first commit")

        repo.upload(
            local_path=dataset_dir,
            remote_path=dataset_dir,
            commit_message=commit_message,
            versioning=versioning,
            last_commit=last_commit,
        )

    def download_dataset(
        self, remote_dir_path: str, local_dir_path: str, branch_name: str
    ) -> None:
        """Download dataset from Dagshub.

        Args:
            remote_dir_path: Path to data on remote
            local_dir_path: Path to data directory locally
            branch_name: Name of branch to use for the dataset.
        """
        self.dagshub_repo.download(
            remote_path=remote_dir_path,
            local_path=local_dir_path,
            recursive=True,
            revision=branch_name,
        )


if __name__ == "__main__":

    dagshub_client = DagsHubClient(owner="dudeperf3ct", name="jetson-data")
    dagshub_client.upload_dataset(
        dataset_dir="results",
        commit_message="Add third version raw dataset",
        branch_name="raw_data",
        versioning="dvc",
    )
