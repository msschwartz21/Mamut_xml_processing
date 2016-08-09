import ast
import importlib
import inspect
from collections import defaultdict

def returnClassMethods(module):
	'''Extracts class and methdos from module and prints with docstring info'''

	data = defaultdict(defaultdict)
	mod = importlib.import_module(module)

	p = ast.parse(inspect.getsource(mod))

	classes = [cls for cls in p.body if isinstance(cls,ast.ClassDef)]

	for cls in classes:

		methods = [node for node in cls.body if isinstance(node,ast.FunctionDef)]

		print('###'+cls.name+'###')
		print(ast.get_docstring(cls))
		for m in methods:
		    print('---')
		    print(m.name + ':')
		    print(ast.get_docstring(m))

def returnModuleFxns(module):
	'''Extracts functions and docstrings from module and prints'''

	data = defaultdict(defaultdict)
	mod = importlib.import_module(module)

	p = ast.parse(inspect.getsource(mod))

	fxns = [node for node in p.body if isinstance(node,ast.FunctionDef)]

	for f in fxns:
		print('---')
		print(f.name+':')
		print(ast.get_docstring(f))