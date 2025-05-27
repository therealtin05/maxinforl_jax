from setuptools import setup, find_packages

required = [
    'jaxrl @ git+https://github.com/sukhijab/jaxrl.git',
    'jaxtyping',
    'numpy==1.26.4',
    'gymnasium==0.29.1',
    'distrax',
    'pyyaml',
]

extras = {}
setup(
    name='maxinforl_jax',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    python_requires='> 3.10',
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
