from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    
    requirements=[]
    try:
        with open(file_path,'r') as file_obj:
            lines=file_obj.readlines()
        
            for line in lines:
                requirement=line.strip()
            
                if requirement and requirement != ". -e":
                    requirements.append(requirement)
    except FileNotFoundError:
        print(f'requirements.txt file not exists')
        
    return requirements
        
        
setup(
    name='InsuranceClaimPredictionProject',
    version='0.0.1',
    author='Sarvesh Chhabra',
    author_email='sarveshpoker@gmail.com',
    packages=find_packages(),
    install_packages=get_requirements('requirements.txt')
)
                
        
    
    