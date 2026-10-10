"""Install the benchmark runtime and its simulator data files."""
from pathlib import Path
from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent
setup(
    name="libero-agent",
    version="0.1.0",
    description="Thirty-task benchmark for agents controlling a LIBERO simulator",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.11",
    url="https://github.com/dzj441/Libero-Agent",
    license="MIT; third-party components retain their respective terms",
    packages=find_namespace_packages(include=["libero", "libero.*", "scripts"]),
    include_package_data=True,
    entry_points={"console_scripts": ["libero-agent=scripts.run_benchmark:main"]},
)
