from pathlib import Path

import pytest


class ProjectPaths:
    def __init__(self) -> None:
        self._current_file_path = Path(__file__)

    @property
    def tests_root(self) -> Path:
        return self._current_file_path.parent

    @property
    def project_root(self) -> Path:
        return self.tests_root.parent

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"


@pytest.fixture
def project_paths() -> ProjectPaths:
    return ProjectPaths()


@pytest.fixture
def dataset_path(project_paths: ProjectPaths) -> Path:
    return project_paths.data_root / "my_dataset"
