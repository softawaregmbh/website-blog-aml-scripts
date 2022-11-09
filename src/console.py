import json
import subprocess
from typing import Union

CONSOLE_COLOR_RESET_CODE = '\x1b[0m'


def execute_cli_command(command: str) -> Union[str, list, dict]:
    command = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, shell=True)
    output, error_output = command.communicate()

    if command.returncode != 0:
        raise RuntimeError(error_output)

    output = output.strip()

    if output.startswith('[') or output.startswith('{'):
        output = output[: -len(CONSOLE_COLOR_RESET_CODE)] if output.endswith(CONSOLE_COLOR_RESET_CODE) else output
        return json.loads(output)
    elif output.startswith('"') and output.endswith('"'):
        return output[1:-1]
    else:
        return output


def start_action(action_text: str):
    print(f'⚪ {action_text}', end='')


def end_action(action_text: str, state: str = 'success'):
    if state == 'success':
        status_symbol = '🟢'
    elif state == 'skipped':
        status_symbol = '🔵'
    elif state == 'failure':
        status_symbol = '🔴'
    else:
        raise ValueError(f'State {state} unhandled.')

    print(f'\r{status_symbol} {action_text}')
