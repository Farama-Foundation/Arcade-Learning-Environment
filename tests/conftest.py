import pytest

import os


class Resources:
    def __init__(self, base_path):
        self.base_path = base_path

    def __getitem__(self, file_name):
        return os.path.join(self.base_path, "resources", file_name)


@pytest.fixture(scope="module")
def resources(request):
    return Resources(request.fspath.dirname)


from fixtures import *
