from setuptools import find_packages, setup

setup(
    name="medibot",
    version="0.0.1",
    author="Vaibhav Narute",
    author_email="narutevaibhav95@gmail.com",
    packages=find_packages(),
    install_requires=[
        'transformers>=4.6.0,<5.0.0',
        'torch>=1.6.0',
        'numpy',
        'scikit-learn',
        'scipy',
        'nltk',
        'huggingface-hub>=0.4.0',
        'pinecone-client[grpc]'
    ],
    python_requires='>=3.8',
)