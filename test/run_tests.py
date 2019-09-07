# Should run all the available tests

import os
import subprocess

from termcolor import colored

tests_list = ['test/base_test.py']

directory = 'test/callbacks_tests'
for fname in os.listdir(directory):
    path = os.path.join(directory, fname)
    tests_list.append(path)

devnull = open(os.devnull, 'w')

failed = []

for test_file in tests_list:
    print("Testing ", test_file, end=' ')
    res = subprocess.call(["python", test_file], stdout=devnull)
    if res:
        print(colored("FAILED", 'red'))
        failed.append(test_file)
    else:
        print(colored("OK", "green"))

if len(failed):
    print(str(len(failed)), "TESTS FAILED:")
    print('\n'.join(failed))
    exit(1)
else:
    print("ALL TESTS PASSED!")


