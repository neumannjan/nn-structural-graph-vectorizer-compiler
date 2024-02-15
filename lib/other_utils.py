import re

REGEX1 = re.compile(r'(.)([A-Z][a-z]+)')
REGEX2 = re.compile(r'([a-z0-9])([A-Z])')

def camel_to_snake(name: str, upper=False):
    name = REGEX1.sub(r'\1_\2', name)
    name = REGEX2.sub(r'\1_\2', name)
    return name.upper() if upper else name.lower()
