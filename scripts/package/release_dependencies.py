"""Include the full relative-import closure, including imports inside functions."""
import ast


def complete_imports(payload, root, seeds=('legalir2/duo.py', 'legalir2/fulltext_retrain.py')):
    pending, seen, added = list(seeds), set(), []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in payload:
            path = root/name
            if not path.is_file():
                raise ValueError('Missing transitive release dependency: '+name)
            payload[name] = path.read_bytes().replace(b'\r\n', b'\n')
            added.append(name)
        for node in ast.walk(ast.parse(payload[name])):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                modules = [node.module] if node.module else [item.name for item in node.names]
                pending.extend('legalir2/'+module.replace('.', '/')+'.py' for module in modules)
    return sorted(added)
