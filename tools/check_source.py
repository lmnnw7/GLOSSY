"""Dependency-free Python, notebook, configuration and documentation checks."""
import ast
import argparse
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_io import load_json_object, read_data_roots, image_roots, validate_raw_meta

EXCLUDED = {'__pycache__', '.ipynb_checkpoints', '.venv'}
IMPORT_ROOTS = (ROOT, ROOT / 'perceptual_backbone')


def local_module(module, directory):
    """Resolve repository imports without importing optional ML dependencies."""
    for base in (directory, *IMPORT_ROOTS):
        target = base.joinpath(*module.split('.'))
        if target.with_suffix('.py').is_file():
            return target.with_suffix('.py')
        if target.is_dir():
            return target
    return None


def check_imports(tree, path):
    failures = []
    local_names = {p.stem for base in IMPORT_ROOTS for p in base.glob('*.py')}
    local_names.update(p.name for base in IMPORT_ROOTS for p in base.iterdir()
                       if p.is_dir() and any(p.rglob('*.py')) and p.name not in EXCLUDED)
    local_names.update(p.stem for p in path.parent.glob('*.py'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules = [node.module]
        else:
            continue
        for module in modules:
            if module.split('.')[0] not in local_names:
                continue
            if local_module(module, path.parent) is None:
                failures.append(f'line {node.lineno}: missing local module {module}')
    return failures


def notebook_tree(source, filename):
    """Check Python cells and the single supported IPython installation magic."""
    lines = []
    for line in source.splitlines(keepends=True):
        if line.startswith('%pip '):
            # This is valid in Jupyter, but not in plain Python source.
            lines.append(f'get_ipython().run_line_magic("pip", {line[5:].strip()!r})\n')
        else:
            lines.append(line)
    source = ''.join(lines)
    compile(source, filename, 'exec')
    return ast.parse(source, filename)


def markdown_anchors(source):
    anchors = set()
    duplicates = {}
    fenced = False
    for line in source.splitlines():
        if line.startswith('```'):
            fenced = not fenced
        if fenced or not re.match(r'^#{1,6} ', line):
            continue
        title = re.sub(r'^#{1,6} ', '', line).strip().lower()
        anchor = re.sub(r'[^\w\s-]', '', title).replace(' ', '-')
        count = duplicates.get(anchor, 0)
        duplicates[anchor] = count + 1
        anchors.add(f'{anchor}-{count}' if count else anchor)
    return anchors


def check_markdown(path):
    source = path.read_text(encoding='utf-8-sig')
    failures = []
    if not source.startswith('# '):
        failures.append('must start with a document title')
    if '**Purpose:**' not in '\n'.join(source.splitlines()[:20]):
        failures.append('must explain its purpose near the title')
    fenced = False
    for number, line in enumerate(source.splitlines(), 1):
        if line.startswith('```'):
            fenced = not fenced
            continue
        if fenced:
            continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', line):
            parsed = urlsplit(target.strip('<>'))
            if parsed.scheme or parsed.netloc:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not destination.exists():
                failures.append(f'line {number}: missing link target {target}')
            elif parsed.fragment and destination.suffix.lower() == '.md':
                if unquote(parsed.fragment) not in markdown_anchors(destination.read_text(encoding='utf-8-sig')):
                    failures.append(f'line {number}: missing heading anchor {target}')
    if fenced:
        failures.append('unclosed code fence')
    return failures


def check_data_paths(roots, field):
    for label, entry in roots.items():
        source = entry.get(field) if isinstance(entry, dict) else entry
        if source is None:
            raise ValueError(f'Dataset {label} must define {field}')
        for root in source if isinstance(source, list) else [source]:
            root = Path(root)
            if root.is_file() and str(root).lower().endswith(('.raw', '.nii.gz')):
                if field == 'latents' and str(root).lower().endswith('.raw'):
                    raise ValueError(f'Latents must use multichannel .nii.gz files, not RAW: {root}')
                continue
            if not root.is_dir():
                raise FileNotFoundError(f'Data path does not exist or is not a volume: {root}')
            files = [path for pattern in ('*.raw', '*.nii.gz') for path in root.glob(pattern) if path.is_file()]
            if not files:
                raise FileNotFoundError(f'No .raw/.nii.gz volumes found in {root}')
            if field == 'latents' and any(str(path).lower().endswith('.raw') for path in files):
                raise ValueError(f'Latents must use multichannel .nii.gz files, not RAW: {root}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data-paths', action='store_true',
                        help='Also require configured datasets to contain real .raw/.nii.gz files')
    args = parser.parse_args()
    failures = []
    files = [p for p in ROOT.rglob('*.py') if not EXCLUDED.intersection(p.parts)]
    for path in files:
        try:
            source = path.read_text(encoding='utf-8-sig')
            compile(source, str(path), 'exec')
            tree = ast.parse(source)
            failures.extend(f'{path.relative_to(ROOT)}: {issue}' for issue in check_imports(tree, path))
        except (SyntaxError, UnicodeError, OSError) as exc:
            failures.append(f'{path.relative_to(ROOT)}: {exc}')
    notebooks = [p for p in ROOT.rglob('*.ipynb') if not EXCLUDED.intersection(p.parts)]
    code_cells = 0
    for path in notebooks:
        try:
            notebook = load_json_object(path)
            if notebook.get('nbformat') != 4 or not isinstance(notebook.get('cells'), list):
                raise ValueError('Expected a version-4 notebook with a cells list')
            for index, cell in enumerate(notebook['cells'], 1):
                if cell['cell_type'] != 'code':
                    continue
                code_cells += 1
                source = cell['source']
                tree = notebook_tree(''.join(source) if isinstance(source, list) else source,
                                     f'{path.name}:cell{index}')
                failures.extend(f'{path.relative_to(ROOT)}:cell{index}: {issue}'
                                for issue in check_imports(tree, path))
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    failures.append(f'{path.relative_to(ROOT)}:cell{index}: clear saved outputs before publishing')
        except (ValueError, KeyError, TypeError, SyntaxError, UnicodeError, OSError) as exc:
            failures.append(f'{path.relative_to(ROOT)}: {exc}')
    configs = list((ROOT/'config').glob('*.json'))
    for path in configs:
        try:
            if path.name == 'raw_meta.example.json':
                validate_raw_meta(load_json_object(path))
            elif path.name in ('REV_VQVAE_data.json', 'SingleRes_dataset.json', 'training_latents.json'):
                numeric = path.name != 'REV_VQVAE_data.json'
                roots = read_data_roots(path, numeric_labels=numeric)
                field = 'latents' if path.name == 'training_latents.json' else 'images'
                if field == 'images':
                    image_roots(path)
                elif any(not isinstance(entry, dict) or 'latents' not in entry for entry in roots.values()):
                    raise ValueError('Exported training manifests must explicitly define latents')
                if args.check_data_paths:
                    check_data_paths(roots, field)
            else:
                load_json_object(path)
        except (ValueError, UnicodeError, OSError) as exc:
            failures.append(f'{path.relative_to(ROOT)}: {exc}')
    markdown = [p for p in ROOT.rglob('*.md') if not EXCLUDED.intersection(p.parts)]
    for path in markdown:
        try:
            failures.extend(f'{path.relative_to(ROOT)}: {issue}' for issue in check_markdown(path))
        except (UnicodeError, OSError, ValueError) as exc:
            failures.append(f'{path.relative_to(ROOT)}: {exc}')
    print(f'Checked {len(files)} Python files, {len(notebooks)} notebooks ({code_cells} code cells), '
          f'{len(configs)} configuration JSON files and {len(markdown)} Markdown files')
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print('Repository checks passed (does not verify dependencies, tensor execution or model quality).')


if __name__ == '__main__':
    main()
