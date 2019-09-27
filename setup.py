import setuptools

setuptools.setup(
    name="vision_ai",
    version="0.0.0",
    description="Artificial intelligence tools and modules for vision tasks.",
    url="https://github.com/ychnlgy/Vision-AI",
    author="Yuchen Li",
    author_email="ychnlgy@utoronto.ca",
    packages=setuptools.find_packages(),
    install_requires=list(map(str.strip, open("requirements.txt"))),
    include_package_data=True
)
