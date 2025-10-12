from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_path.readlines()
        requirements=[req.replace("/n","") for req in requirements]
    return requirements

setup(
name="ML project",
version='0.0.1',
author='Aparna',
packages=find_packages()
intall_reqiure=get_requirements('requirements.txt')
)