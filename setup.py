from setuptools import setup, find_packages

setup(
    name='cape',
    packages=find_packages(),
    version='latest',
    description='Continuous Augmented Positional Embeddings',
    author='Guillermo Cambara',
    author_email='guillermocambara@gmail.com',
    url='https://github.com/gcambara/cape',
    install_requires=[
        'torch>=1.10.0',
    ],
    keywords=['continuous_augmented_positional_embeddings', 'cape', 'positional_embeddings',
              'positional_encodings', 'transformer'],
    python_requires='>=3.6'
)
