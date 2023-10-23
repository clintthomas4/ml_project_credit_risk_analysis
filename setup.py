from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str):
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Remove leading/trailing whitespace

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='ml_capstone1',
    version='0.0.1',
    author='Clint',
    author_email='clintthomas65@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)