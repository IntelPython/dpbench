import sys, string, os, re, subprocess, time

def mydate():
    s = time.strftime("%b:%d:%Y:%T", time.localtime(time.time()))
    s = string.replace(s, ":", "-")
    return s

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
        print("Creating directory `%s`" % path)
        os.mkdir(path)

def chdir(path):
    if os.path.exists(path):
        print("Changing to directory `%s`" % path)
        os.chdir(path)
        return True
    else:
        return False

########## Log file code ##########

message_log_string = ""
error_counter = 0

def log_message(s="", newline=True):
    global message_log_string
    if newline:
        s += "\n"
    message_log_string += s
    sys.stdout.write(s)
    sys.stdout.flush()
    
def log_error(s=""):
    global error_counter
    log_message(s)
    error_counter += 1
    
def log_heading(s="", character="-"):
    log_message()
    log_message(s)
    log_message(character*len(s))
    log_message()

########## run command code ##########

class ExperimentError(Exception):
    def __init__(self, command, output):
        self.command = command
        limit = 10000
        if(len(output) > limit):
          self.output = output[:limit/2] + "\n\n...TRUNCATED...\n\n" + output[-limit/2:]
        else:
          self.output = output
    def __str__(self):
        return "ExperimentError:" + repr(self.command)

def run_command(command_string, input_string="", max_lines=0, verbose=False, echo=True, throw_exception=True, filename=None):

    if echo:
        print("executing:", command_string)
    #obj = popen2.Popen4(command_string)

    try:
        output = subprocess.check_output(command_string, text=True)
        #output = subprocess.check_output(["echo", "hello world"])

        if verbose == 1:
            if filename is None:
                print(output)
            else:
                with open(filename, 'w', 1) as fd:
                    fd.write(output)
    except subprocess.CalledProcessError as e:
        if throw_exception:
            raise ExperimentError(command_string, e.output)

    # ferr.seek(0)
    # error = ferr.read()

    # if error and throw_exception:

    # fout.seek(0)

    # output = fout.readline()


    # obj = subprocess.Popen(command_string)
    # output = ""

    # obj.tochild.write(input_string)
    # obj.tochild.close()
    # line = obj.fromchild.readline()
    # while (line):
    #     if verbose == 1:
    #         print(line,)
    #     output += line
    #     line = obj.fromchild.readline()
    # exit_status = obj.wait()

    # if(max_lines != 0):
    #     lines = output.split("\n");
    #     output = string.join(lines[-max_lines:], "\n")

    # if throw_exception and exit_status != 0:
    #     raise ExperimentError(command_string, output)

    return output

########## main code ##########
