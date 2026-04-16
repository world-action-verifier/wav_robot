import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="udrm",
    version="0.0.0",
    description="IDM (minimal CLAM-based training stack)",
    url="",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="CLAM Robots",
    packages=find_packages(exclude=["tests", ".github"]),
    # requirements.txt is not shipped; leave empty so editable install succeeds after deps are preinstalled.
    install_requires=[],
)
