from setuptools import setup, find_packages

setup(
    name='pygdgen',
    version='1.0.0',
    author='NingWang',
    author_email='a320873529@gmail.com',
    description=('PyGDGen is a versatile tool for processing multiple and '
                 'customized cluster configurations. It excels in generating '
                 'dense configurations and allows users to set customized '
                 'constraint boxes, like cubes with periodic conditions, '
                 'neck-like structures, or non-periodic environments.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NingWang-art/PyGDGen',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7',
        'numpy>=1.21',
        'scipy>=1.7',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
