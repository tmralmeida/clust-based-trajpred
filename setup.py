from setuptools import setup, find_packages

setup(
    name="motion_pred_clustbased",
    version="0.1.0",
    description="Trajectory prediction with clustering based methods",
    license="MIT",
    author="Tiago Rodrigues de Almeida",
    author_email="tmr.almeida96@gmail.com",
    python_requires="==3.9.13",
    url="https://github.com/tmralmeida/motion-pred-clustbased",
    packages=find_packages(include=['motion_pred_clustbased', 'motion_pred_clustbased.*'])
)
