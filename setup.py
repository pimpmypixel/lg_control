from setuptools import setup, find_packages

setup(
    name="smart-tv-controller",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiowebostv==0.1.3",
        "asyncio>=3.4.3",
        "greenlet==3.0.1",
        "numpy==1.24.4",
        "opencv-python==4.8.1.78",
        "pillow==10.4.0",
        "playwright==1.40.0",
        "pyee==11.0.1",
        "pylgtv==0.1.9",
        "python-dotenv==1.0.0",
        "typing_extensions==4.13.2",
        "websockets>=12.0",
    ],
    python_requires=">=3.10",
    author="PimpMyPixel",
    author_email="your.email@example.com",
    description="Smart TV Streaming Controller",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 