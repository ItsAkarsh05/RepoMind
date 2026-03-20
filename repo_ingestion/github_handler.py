"""
GitHub Repository Handler
=========================
Validates GitHub URLs and clones repositories locally with caching/overwrite support.
"""

import os
import re
import shutil
import subprocess
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL Validation
# ---------------------------------------------------------------------------

def validate_github_url(url: str) -> tuple[str, str]:
    """
    Validate a public GitHub repository URL and extract owner + repo name.

    Supports formats:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - https://github.com/owner/repo/  (trailing slash)

    Args:
        url: The GitHub URL to validate.

    Returns:
        A tuple of (owner, repo_name).

    Raises:
        ValueError: If the URL does not match the expected GitHub pattern.
    """
    if not url or not isinstance(url, str):
        raise ValueError("GitHub URL must be a non-empty string.")

    url = url.strip().rstrip("/")

    pattern = r"^https://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?$"
    match = re.match(pattern, url)

    if not match:
        raise ValueError(
            f"Invalid GitHub URL: '{url}'. "
            "Expected format: https://github.com/<owner>/<repo>"
        )

    owner, repo = match.group(1), match.group(2)
    return owner, repo


# ---------------------------------------------------------------------------
# Repository Cloning
# ---------------------------------------------------------------------------

def clone_repository(url: str, clone_dir: str = "cloned_repos") -> str:
    """
    Clone a GitHub repository into a local directory.

    If the target directory already exists it is removed first so the latest
    version of the repo is always used — this prevents stale data and avoids
    duplicate embeddings.

    Args:
        url:       The HTTPS GitHub repository URL.
        clone_dir: Parent directory where repos are cloned into.
                   Each repo gets its own sub-directory named after the repo.

    Returns:
        The absolute path to the cloned repository.

    Raises:
        ValueError:      If the URL is invalid.
        RuntimeError:     If the ``git clone`` command fails.
    """
    owner, repo = validate_github_url(url)

    # Ensure the parent clone directory exists
    os.makedirs(clone_dir, exist_ok=True)

    repo_path = os.path.join(clone_dir, repo)

    # Remove existing clone to guarantee fresh data
    if os.path.exists(repo_path):
        logger.info("Removing existing clone at '%s' to re-clone fresh data.", repo_path)
        shutil.rmtree(repo_path)

    logger.info("Cloning '%s/%s' into '%s' …", owner, repo, repo_path)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, repo_path],
            check=True,
            text=True,
            capture_output=True,
        )
        logger.debug("git clone stdout: %s", result.stdout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to clone repository '{url}': {exc.stderr}"
        ) from exc

    return os.path.abspath(repo_path)
