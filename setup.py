from pathlib import Path
from setuptools import setup, find_packages

def readme():
    readme_path = Path(__file__).parent / "README.md"
    return readme_path.read_text(encoding="utf-8")

setup(
    name="immuscope",
    version="0.1.0",
    description="ImmuScope: MHC-II epitope immunogenicity prediction",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/griffithlab/ImmuScope",
    author="Shen Long-Chen, Zhang Yumeng, Wang Zhikang, Littler Dene R, Liu Yan, Tang Jinhui, Rossjohn Jamie, Yu Dong-Jun, Song Jiangning",
    license="GPL-3.0-only",
    packages=find_packages(exclude=("tests*", "weights*", "configs*")),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.12",
    ],
    project_urls={
        "Upstream": "https://github.com/shenlongchen/ImmuScope",
        "Source": "https://github.com/griffithlab/ImmuScope",
        "Issues": "https://github.com/griffithlab/ImmuScope/issues",
    },
    zip_safe=False
)