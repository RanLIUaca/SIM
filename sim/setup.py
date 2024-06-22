from setuptools import find_packages, setup

def read_lines(f):
    with open(f) as in_f:
        r = in_f.readlines()
    r = [l.strip() for l in r]
    return r



description = "SIM: Probabilistic Generative Model for Sequence Interaction with Application to TCR-peptide Binding"

install_requires = read_lines("./requirements.txt")

setup(
    name='sim',
    version='0.0.1',
    description=description,
    packages=find_packages(),
    install_requires=install_requires,
    # url='https://github.com/ranliuaca/...',
    author='Ran Liu',
    author_email='rliu@link.cuhk.edu.hk',
    license='MIT',
)