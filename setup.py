from setuptools import setup

def readme():
	with open('README.rst') as f:
		return (f.read())

def requirements():
	with open('requirements.txt') as f:
		return (f.read())

setup(name='mxp',
	version='0.0.0',
	description='',
	long_description=readme(),
	keywords='',
	url='https://github.com/msschwartz21/Mamut_xml_processing',
	author='Morgan Schwartz',
	author_email='msschwartz21@gmail.com',
	license='MIT',
	packages=['mxp'],
	install_requires=[])