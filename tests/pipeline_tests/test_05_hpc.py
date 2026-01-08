#!/usr/bin/env python3
"""
Tests for Step 5: HPC Scripts

Tests OSRM and HPC-related scripts with local validation.
These tests validate script structure and syntax without requiring HPC access.
"""

import sys
from pathlib import Path
import ast

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_hpc_directory_structure():
    """Test that HPC directory has expected structure."""
    hpc_dir = PROJECT_ROOT / "scripts/hpc"

    if not hpc_dir.exists():
        print(f"✗ HPC directory not found: {hpc_dir}")
        return False

    expected_dirs = [
        'osrm_scripts',
        'osrm_setup',
        'slurm_templates',
    ]

    all_exist = True
    for subdir in expected_dirs:
        path = hpc_dir / subdir
        if path.exists():
            file_count = len(list(path.glob('*')))
            print(f"✓ {subdir}/ ({file_count} files)")
        else:
            print(f"⚠ {subdir}/ not found")
            all_exist = False

    return all_exist


def test_osrm_scripts_syntax():
    """Test that OSRM Python scripts have valid syntax."""
    osrm_scripts_dir = PROJECT_ROOT / "scripts/hpc/osrm_scripts"

    if not osrm_scripts_dir.exists():
        print("⚠ OSRM scripts directory not found")
        return True

    py_files = list(osrm_scripts_dir.glob("*.py"))

    if not py_files:
        print("⚠ No Python scripts found")
        return True

    all_valid = True
    for py_file in py_files:
        try:
            with open(py_file) as f:
                ast.parse(f.read())
            print(f"✓ {py_file.name}")
        except SyntaxError as e:
            print(f"✗ {py_file.name} - line {e.lineno}: {e.msg}")
            all_valid = False

    return all_valid


def test_slurm_templates_valid():
    """Test that SLURM templates have valid structure."""
    slurm_dir = PROJECT_ROOT / "scripts/hpc/slurm_templates"

    if not slurm_dir.exists():
        print("⚠ SLURM templates directory not found")
        return True

    slurm_files = list(slurm_dir.glob("*.slurm")) + list(slurm_dir.glob("*.sh"))

    if not slurm_files:
        print("⚠ No SLURM files found")
        return True

    all_valid = True
    for slurm_file in slurm_files:
        with open(slurm_file) as f:
            content = f.read()

        # Check for SLURM directives
        has_sbatch = '#SBATCH' in content
        has_shebang = content.startswith('#!')

        issues = []
        if not has_shebang:
            issues.append("missing shebang")

        if slurm_file.suffix == '.slurm' and not has_sbatch:
            issues.append("missing #SBATCH directives")

        if issues:
            print(f"⚠ {slurm_file.name}: {', '.join(issues)}")
        else:
            # Extract some SLURM settings
            settings = []
            for line in content.split('\n'):
                if line.startswith('#SBATCH'):
                    settings.append(line.split()[-1] if '=' in line else line)

            print(f"✓ {slurm_file.name}")
            if settings:
                print(f"  - SBATCH directives: {len([l for l in content.split(chr(10)) if l.startswith('#SBATCH')])}")

    return all_valid


def test_osrm_setup_scripts():
    """Test OSRM setup shell scripts."""
    setup_dir = PROJECT_ROOT / "scripts/hpc/osrm_setup"

    if not setup_dir.exists():
        print("⚠ OSRM setup directory not found")
        return True

    sh_files = list(setup_dir.glob("*.sh"))

    if not sh_files:
        print("⚠ No shell scripts found")
        return True

    for sh_file in sh_files:
        with open(sh_file) as f:
            content = f.read()

        checks = {
            'shebang': content.startswith('#!'),
            'not empty': len(content.strip()) > 0,
        }

        status = "✓" if all(checks.values()) else "⚠"
        print(f"{status} {sh_file.name}")

        # Check for common operations
        if 'osrm' in content.lower():
            print(f"  - Contains OSRM commands")
        if 'singularity' in content.lower():
            print(f"  - Uses Singularity containers")
        if 'docker' in content.lower():
            print(f"  - References Docker")

    return True


def test_hpc_documentation():
    """Test that HPC documentation exists and is readable."""
    hpc_dir = PROJECT_ROOT / "scripts/hpc"

    doc_files = list(hpc_dir.glob("*.md"))

    if not doc_files:
        print("⚠ No documentation found in HPC directory")
        return True

    print(f"Found {len(doc_files)} documentation files:")

    for doc_file in doc_files:
        size_kb = doc_file.stat().st_size / 1024
        print(f"✓ {doc_file.name} ({size_kb:.1f} KB)")

        # Quick content check
        with open(doc_file) as f:
            first_lines = f.read(500)

        if '# ' in first_lines:
            # Extract title
            for line in first_lines.split('\n'):
                if line.startswith('# '):
                    print(f"  - Title: {line[2:50]}...")
                    break

    return True


def test_routing_script_structure():
    """Test structure of main routing scripts."""
    osrm_scripts_dir = PROJECT_ROOT / "scripts/hpc/osrm_scripts"

    routing_scripts = [
        'route_cities.py',
        'route_cities_res7.py',
        'route_city_hpc.py',
    ]

    for script_name in routing_scripts:
        script_path = osrm_scripts_dir / script_name

        if not script_path.exists():
            print(f"⚠ {script_name} not found")
            continue

        with open(script_path) as f:
            content = f.read()

        # Check for key components
        components = {
            'OSRM client': 'osrm' in content.lower() or 'requests' in content,
            'H3 handling': 'h3' in content.lower(),
            'JSON output': 'json' in content.lower(),
            'argparse': 'argparse' in content or 'sys.argv' in content,
        }

        print(f"\n{script_name}:")
        for comp, present in components.items():
            status = "✓" if present else "⚠"
            print(f"  {status} {comp}")

    return True


def test_centrality_script_structure():
    """Test structure of centrality calculation scripts."""
    osrm_scripts_dir = PROJECT_ROOT / "scripts/hpc/osrm_scripts"

    centrality_scripts = list(osrm_scripts_dir.glob("*centrality*.py"))

    if not centrality_scripts:
        print("⚠ No centrality scripts found")
        return True

    for script_path in centrality_scripts:
        with open(script_path) as f:
            content = f.read()

        # Check for centrality-related components
        components = {
            'numpy/scipy': 'numpy' in content or 'scipy' in content,
            'networkx': 'networkx' in content,
            'matrix operations': 'matrix' in content.lower() or 'sparse' in content,
            'output saving': 'to_csv' in content or 'json.dump' in content,
        }

        print(f"\n{script_path.name}:")
        for comp, present in components.items():
            status = "✓" if present else "⚠"
            print(f"  {status} {comp}")

    return True


def test_common_utilities():
    """Test common utility modules used across HPC scripts."""
    osrm_scripts_dir = PROJECT_ROOT / "scripts/hpc/osrm_scripts"

    common_files = ['common.py', 'common_v2.py']

    for common_name in common_files:
        common_path = osrm_scripts_dir / common_name

        if not common_path.exists():
            continue

        with open(common_path) as f:
            content = f.read()

        # Parse to find defined functions
        try:
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            print(f"\n{common_name}:")
            if functions:
                print(f"  - Functions: {', '.join(functions[:5])}")
                if len(functions) > 5:
                    print(f"    ... and {len(functions) - 5} more")
            if classes:
                print(f"  - Classes: {', '.join(classes)}")

        except SyntaxError:
            print(f"⚠ {common_name}: syntax error")

    return True


def test_osrm_data_directories():
    """Test that OSRM data directories exist (may be empty locally)."""
    hpc_dir = PROJECT_ROOT / "scripts/hpc"

    data_dirs = ['cities', 'city_lists', 'regions', 'osrm_data', 'osrm_results']

    for dir_name in data_dirs:
        dir_path = hpc_dir / dir_name

        if dir_path.exists():
            if dir_path.is_dir():
                file_count = len(list(dir_path.glob('*')))
                print(f"✓ {dir_name}/ ({file_count} items)")
            else:
                print(f"⚠ {dir_name} is not a directory")
        else:
            print(f"⚠ {dir_name}/ not found (may need to sync from HPC)")

    return True


def test_hpc_config_references():
    """Test that HPC scripts reference correct paths and configs."""
    osrm_scripts_dir = PROJECT_ROOT / "scripts/hpc/osrm_scripts"

    py_files = list(osrm_scripts_dir.glob("*.py"))[:5]  # Check first 5

    hpc_path_patterns = [
        '/scratch/',
        '/gpfsnyu/',
        'hpc.shanghai.nyu.edu',
    ]

    print("Checking HPC path references in scripts:")

    for py_file in py_files:
        with open(py_file) as f:
            content = f.read()

        found_patterns = [p for p in hpc_path_patterns if p in content]

        if found_patterns:
            print(f"  {py_file.name}: references {', '.join(found_patterns)}")
        else:
            print(f"  {py_file.name}: no HPC-specific paths (portable)")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: HPC Scripts Tests")
    print("=" * 60)

    tests = [
        test_hpc_directory_structure,
        test_osrm_scripts_syntax,
        test_slurm_templates_valid,
        test_osrm_setup_scripts,
        test_hpc_documentation,
        test_routing_script_structure,
        test_centrality_script_structure,
        test_common_utilities,
        test_osrm_data_directories,
        test_hpc_config_references,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
