import re
from setuptools import find_packages, setup

try:
    import pypandoc
    readme = pypandoc.convert('README.md', 'rst')
    history = pypandoc.convert('CHANGELOG.md', 'rst')
except (ImportError, OSError):
    with open('README.md') as readme_file, \
            open('CHANGELOG.md') as history_file:
        readme = readme_file.read()
        history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

with open('requirements-dev.txt') as dev_requirements_file:
    dev_requirements = dev_requirements_file.read().splitlines()

with open('requirements-test.txt') as test_requirements_file:
    test_requirements = test_requirements_file.read().splitlines()
    dev_requirements.extend(test_requirements)

version_regex = re.compile(r'__version__ = [\'\"]v((\d+\.?)+)[\'\"]')
with open('src/simplenet/__init__.py') as f:
    vlines = f.readlines()
__version__ = next(re.match(version_regex, line).group(1) for line in vlines
                   if re.match(version_regex, line))

setup(
    name="simplenet",
    version=__version__,
    description="A simple neural network in Python",
    long_description=readme + "\n\n" + history,
    author="Nathan Henrie",
    author_email="nate@n8henrie.com",
    url="https://github.com/n8henrie/simplenet",
    packages=find_packages("src"),
    include_package_data=True,
    package_dir={"": "src"},
    # entry_points={
    #     'console_scripts': ['simplenet=simplenet.cli:run']
    #     },
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords="simplenet",
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6"
    ],
    extras_require={
        "dev": dev_requirements
        },
    test_suite="tests",
    tests_require=test_requirements,
    #python_requires=">=3.6",
)
