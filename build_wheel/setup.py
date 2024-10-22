from setuptools import setup, find_packages
# from Cython.Build import cythonize


setup(
    name="yolov8_onnx_to_tensorrt",
    version="1.0.0",
    description="YOLOv8 ONNX to TensorRT Conversion",
    
    # ext_modules=cythonize("yolov8_tensorrt/**/*.py", compiler_directives={'language_level': "3"}),  # 使用 Cython 编译所有 .py 文件,
    packages=find_packages(),
    install_requires=[
        # 'numpy<=1.23.5',
        # 'onnx',
        # 'onnxsim',
        # 'opencv-python',
        # 'torch',
        # 'torchvision',
        # 'ultralytics',
    ],
    entry_points={
        'console_scripts': [
            'yolov8_onnx_convert_trt=yolov8_onnx_to_tensorrt.build:main',  # 创建命令行工具
        ],
    },
    include_package_data=True,
)