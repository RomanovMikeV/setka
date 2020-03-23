# Should run all the available tests

import os
import sys
import gc
import traceback

from termcolor import colored
from subprocess import Popen, PIPE

# Disable

def run_command(command):
    if not isinstance(command, (list, tuple)):
        command = command.split(' ')

    process = Popen(command, stdout=PIPE)
    output, err = process.communicate()
    exit_code = process.wait()
    return exit_code, err, output.decode()


tests_list = ['python test/base_test.py']

directories = ['test/pipes_tests']

for directory in directories:
    for fname in os.listdir(directory):
        if fname[:2] != '__':
            tests_list.append('python ' + os.path.join(directory, fname))

# devnull = open(os.devnull, 'w')

failed = []



for test_file in tests_list:
    print("Testing ", test_file, end=' ')

    res = run_command(test_file)
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


