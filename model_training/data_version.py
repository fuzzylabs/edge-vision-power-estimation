"""Dagshub Client."""

import argparse
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
    parser = argparse.ArgumentParser(
        description="Use DagsHub and DVC to version control datasets"
    )
    parser.add_argument(
        "--owner",
        type=str,
        default="fuzzylabs",
        help="Name of user/organization on DagsHub.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="edge-vision-power-estimation",
        help="The directory to save the log result.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="Add dataset to DagsHub",
        help="Commit message",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Name of branch to push data",
    )
    parser.add_argument(
        "--local-dir-path",
        type=str,
        default="raw_data",
        help="The local directory to version control using DVC and DagsHub.",
    )
    parser.add_argument(
        "--remote-dir-path",
        type=str,
        default="raw_data",
        help="The remote directory to pull from DagsHub.",
    )
    parser.add_argument(
        "--versioning",
        type=str,
        default="dvc",
        help="Which versioning system to use to upload a file.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Push data to DagsHub",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pull data from DagsHub",
    )
    args = parser.parse_args()

    dagshub_client = DagsHubClient(owner=args.owner, name=args.name)
    if args.upload:
        print(f"Pushing {args.local_dir_path} directory to DagsHub")
        dagshub_client.upload_dataset(
            dataset_dir=args.local_dir_path,
            commit_message=args.commit,
            branch_name=args.branch,
            versioning=args.versioning,
        )
    if args.download:
        dagshub_client.download_dataset(
            remote_dir_path=args.remote_dir_path,
            local_dir_path=args.local_dir_path,
            branch_name=args.branch,
        )
