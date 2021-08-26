import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#############################################################
#############################################################


PACKAGE_NAME = 'vgtk'
EXT_MODULES = ['gathering', 'grouping', 'zpconv']
PACKAGES = ['app', 'cuda', 'functional', 'point3d', 'pc', 'mesh', 'voxel', 'spconv', 'so3conv', 'transform', 'data.anchors']
INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 
                        'scikit-image', 'tqdm', 'imageio', 'plyfile',
                        'parse', 'colour']


#############################################################
#############################################################


def cuda_extension(package_name, ext):
    ext_name = f"{package_name}.cuda.{ext}"
    ext_cpp = f"{package_name}/cuda/{ext}_cuda.cpp"
    ext_cu = f"{package_name}/cuda/{ext}_cuda_kernel.cu"
    return CUDAExtension(ext_name, [ext_cpp, ext_cu])


pkg_name = PACKAGE_NAME
ext_modules = [cuda_extension(pkg_name, ext) for ext in EXT_MODULES]
pkgs = [pkg_name] + [f"{pkg_name}.{pkg}" for pkg in PACKAGES]
install_reqs = [req for req in INSTALL_REQUIREMENTS]


setup(
    description='Vision-Graphics deep learning ToolKit',
    author='VGL (Shichen Liu*, Haiwei Chen*)',
    author_email='liushichen95@gmail.com',
    license='MIT License',
    version='0.0.1',
    name=pkg_name,
    packages=pkgs,
    package_data={'':['*.ply']},
    include_package_data=True,
    install_requires=install_reqs,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)