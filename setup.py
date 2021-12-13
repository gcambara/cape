from setuptools import setup, find_packages

setup(
    name='cape',
    packages=find_packages(),
    version='1.0.0',
    license='MIT',
    description='Continuous Augmented Positional Embeddings for PyTorch',
    author='Guillermo Cámbara',
    author_email='guillermocambara@gmail.com',
    url='https://github.com/gcambara/cape',
    download_url='https://github.com/gcambara/cape/archive/v_100.tar.gz',
    install_requires=[
        'einops>=0.3.2',
        'torch>=1.10.0',
    ],
    keywords=['continuous_augmented_positional_embeddings', 'cape', 'positional_embeddings',
              'positional_encodings', 'transformer'],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
