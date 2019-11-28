# Should run all the available tests

import os
import sys
import gc
import subprocess

from termcolor import colored

# Disable

def run_python_script(script_path):
    script_name = script_path.split('/')[-1]
    script_dir = '/'.join(script_path.split('/')[:-1])
    script_name = script_name[:-3]

    sys.path.append(script_dir)

    sys.stdout = open(os.devnull, 'w')
    exit_code = 0
    message = None

    try:
        test = __import__(script_name)
        del test
    except Exception as e:
        exit_code = 1
        message = str(e)

    gc.collect()
    sys.stdout = sys.__stdout__
    return exit_code, message



tests_list = ['test/base_test.py']

directory = 'test/pipes_tests'
# sys.path.append(directory)
# sys.path.append('test')

for fname in os.listdir(directory):
    if fname[:2] != '__':
        tests_list.append(os.path.join(directory, fname))

# devnull = open(os.devnull, 'w')

failed = []



for test_file in tests_list:
    print("Testing ", test_file, end=' ')

    res = run_python_script(test_file)
    if res[0]:
        print(colored("FAILED", 'red'))
        print(res[1])
        failed.append(test_file)
    else:
        print(colored("OK", "green"))


if len(failed):
    print(str(len(failed)), "TESTS", colored("FAILED", "red"), ":")
    print('\n'.join(failed))
    exit(1)
else:
    print("TESTS", colored("PASSED", 'green'), "!")


