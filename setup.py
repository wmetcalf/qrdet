from setuptools import setup, find_namespace_packages
import os

setup(
    name='qrdet',
    version='2.5',
    author_email='eric@ericcanas.com',
    author='Eric Canas',
    url='https://github.com/Eric-Canas/qrdet',
    description='Robust QR Detector based on YOLOv8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(include=['qrdet', 'qrdet.*']),  # Specify package names explicitly
    license='MIT',
    include_package_data=True,  # Ensures non-Python files are included
    package_data={
        'qrdet': ['/models/*.onnx'],  # Ensure the models are in the 'qrdet' package folder
    },
    install_requires=[
        'quadrilateral-fitter',
        'numpy',
        'onnxruntime'    
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Topic :: Multimedia :: Graphics',
        'Typing :: Typed',
    ],
)

