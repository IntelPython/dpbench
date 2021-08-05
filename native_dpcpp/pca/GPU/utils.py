# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import re
import os
import subprocess
import sys
import time


def mydate():
    return time.strftime('%b-%d-%Y-%H-%M-%S', time.localtime(time.time()))


def parse_time(s):
    regex = r"""
    ([\d.]*)user [\s]*
    ([\d.]*)system [\s]*
    ([\d.:]*)elapsed[\s]*
    [\d.]* [%]CPU [\s]*
    """

    obj = re.compile(regex, re.VERBOSE).search(s)
    assert(obj)
    
    user, system, elapsed = obj.groups()
    return float(user)


########## utility functions ##########


def mkdir(path):
    if not os.path.exists(path):
        print('Creating directory `{}`'.format(path))
        os.mkdir(path)


def chdir(path):
    if os.path.exists(path):
        print('Changing to directory `{}`'.format(path))
        os.chdir(path)


########## Log file code ##########


message_log_string = ''
error_counter = 0


def log_message(s='', newline=True):
    global message_log_string
    if newline:
        s += '\n'
    message_log_string += s
    sys.stdout.write(s)
    sys.stdout.flush()


def log_error(s=''):
    global error_counter
    log_message(s)
    error_counter += 1


def log_heading(s='', character='-'):
    log_message()
    log_message(s)
    log_message(character*len(s))
    log_message()


########## run command code ##########


class ExperimentError(Exception):
    def __init__(self, command, output):
        self.command = command
        limit = 10000
        if len(output) > limit:
            self.output = output[:limit/2] + '\n\n...TRUNCATED...\n\n' + output[-limit/2:]
        else:
            self.output = output

    def __str__(self):
        return 'ExperimentError:' + repr(self.command)


def run_command(command_string, verbose=False, echo=True, throw_exception=True):

    if echo:
        print('executing:', subprocess.list2cmdline(command_string))

    try:
        output = subprocess.check_output(command_string)

        if verbose == 1:
            print(output)
    except subprocess.CalledProcessError as e:
        if throw_exception:
            raise ExperimentError(command_string, e.output)
    else:
        return output
