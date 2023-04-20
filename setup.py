from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        # Remove any editable requirement specified with '-e.' at the beginning of the line

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements
    
    
setup(
    name='ml-best-practices',
    version='0.0.1',
    author='Karan Shingde',
    author_email='karanshingde@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)