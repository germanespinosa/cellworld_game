#!/bin/python3

import sys
from easy_pack import EasyPackModule
from os import path

module = EasyPackModule.read('.')
if not path.exists('setup/setup.py') or path.getctime('__info__.py') > path.getctime('setup/setup.py'):
    print('package info file has changed, rebuilding setup')

build = module.build_module('python-build')
if build:
    print('build succeded')
    module.save('.')
    if '-upload' in sys.argv:
        import os
        username = ""
        if '-user' in sys.argv:
            username = sys.argv[sys.argv.index('-user') + 1]
        password = ""
        if '-password' in sys.argv:
            password = sys.argv[sys.argv.index('-password') + 1]
        repository = ""
        if '-repository' in sys.argv:
            repository = sys.argv[sys.argv.index('-repository')  + 1]
        cur_dir = os.getcwd()
        os.chdir(build)
        upload_command = 'python -m twine upload dist/*' + ((' --repository-url  ' + repository) if repository else '') + ((' -u ' + username) if username else '') + ((' -p ' + password) if password else '')
        os.system(upload_command)
        os.chdir(cur_dir)
    else:
        print('use twine upload --repository-url [pypi-repository-url] dist/* to upload the package')
    if '-install' in sys.argv:
        import os
        cur_dir = os.getcwd()
        os.chdir(build)
        os.system('pip install .')
        os.chdir(cur_dir)
else:
    print('build failed')
