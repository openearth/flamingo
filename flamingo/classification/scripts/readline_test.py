import readline
COMMANDS = ['extra', 'extension', 'stuff', 'errors',
            'email', 'foobar', 'foo']

def complete(text, state):
    for cmd in COMMANDS:
        if cmd.startswith(text):
            if not state:
                return cmd
            else:
                state -= 1

if 'libedit' in readline.__doc__:
    readline.parse_and_bind("bind ^I rl_complete")
else:
    readline.parse_and_bind("tab: complete")
readline.set_completer(complete)
raw_input('Enter section name: ')
