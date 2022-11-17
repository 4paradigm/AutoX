# -*- coding: utf-8 -*-
from distutils.core import setup
from setuptools import find_packages

setup(name="automl-x",
        version="0.3.0",
        description="automl tools",
        author="caihengxing",
        author_email="caihengxing@4paradigm.com",
        url='https://github.com/4paradigm/autox',

        install_requires=[
            'lightgbm==2.1.0',
            'xgboost==1.5.0',
            # 'pytorch-tabnet',
            'torch==1.10.0',
            'numpy==1.21.4',
            'pandas==1.3.5',
            'scikit-learn==0.24.0',
            'tqdm==4.62.3',
            'optuna==2.10.0',
            'img2vec_pytorch==0.2.5',
            'pypinyin==0.44.0',
            'keras==2.7.0',
            'tensorflow==2.7.0',
            'gensim==4.2.0',
            # 'glove-python-binary',
            # 'transformers',
            # 'datasets',
            # 'pyarrow>=6.0.0',
        ],
        python_requires='>=3.6',
        packages=find_packages(exclude=['data', 'demo', 'img', 'sub', 'test',
                                        'run.py', 'run_oneclick.py', 'submit.py']),
        include_package_data=True,
        zip_safe=False
        )
# python setup.py sdist build
# twine upload dist/*
