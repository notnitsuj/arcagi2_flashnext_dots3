# Prompt template and answer parsers from github.com/arcprize/arc-agi-benchmarking
# (MIT License, Copyright (c) 2025 ARC Prize Foundation).
import json, re
TEMPLATE = 'You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--\n{training_examples}\n--End of Training Examples--\n\n--Test Input--\n{test_input}\n--End of Test Input--\n\nYour response:'

def make_prompt(task, test_idx):
    ex = ''
    for i, p in enumerate(task['train']):
        ex += f'--Example {i}-- \n\n INPUT: \n\n' + json.dumps(p['input']) + '\n\n'
        ex += 'OUTPUT: \n\n' + json.dumps(p['output']) + '\n\n'
    return TEMPLATE.format(training_examples=ex, test_input=json.dumps(task['test'][test_idx]['input']))

def backscan_json_parser(log_str):
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(log_str) - 1, -1, -1):
        char = log_str[i]
        if char in (']', '}'):
            last_bracket_idx = i
            closing_bracket = char
            break
    if last_bracket_idx == -1:
        return None
    opening_bracket = '[' if closing_bracket == ']' else '{'
    bracket_counter = 1
    start_idx = -1
    for i in range(last_bracket_idx - 1, -1, -1):
        char = log_str[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break
    if start_idx == -1:
        return None
    json_candidate = log_str[start_idx:last_bracket_idx + 1]
    try:
        parsed_json = json.loads(json_candidate)
        if isinstance(parsed_json, list) and parsed_json and all((isinstance(row, list) for row in parsed_json)):
            return parsed_json
        else:
            return None
    except json.JSONDecodeError:
        return None

def extract_from_boxed(log_str):
    match = re.search('\\\\boxed\\{(.*?)\\}', log_str, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all((isinstance(i, list) for i in parsed_json)):
                return parsed_json
        except json.JSONDecodeError:
            pass
    return None

def parse(answer):
    for f in (extract_from_boxed, backscan_json_parser):
        try:
            r = f(answer)
        except Exception:
            r = None
        if r is not None:
            return r
    return None
