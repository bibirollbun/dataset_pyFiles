


#!/usr/bin/env python3

import os
import sys
import json
import copy
import re
import ast
import zipfile
from pathlib import Path
from datetime import datetime
from itertools import combinations, product

sys.path.insert(0, r'/kaggle/input/google-code-golf-2025\code_golf_utils')

class Optimizer:
    """ optimization with compressed file handling"""

    def __init__(self):
        self.data_dir = Path(r'/kaggle/input/google-code-golf-2025')
        self.baseline_dir = Path(r'/kaggle/input/r30-neurips-golf-lessons-learned/submission')
        self.output_dir = Path('submission_')

        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir()

        self.improvements = []
        self.start_time = datetime.now()

        # Prioritize longest tasks
        self.task_sizes = {}

    def log(self, message, level=0):
        """Print with timestamp and indentation"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        indent = "  " * level
        print(f"[{elapsed:7.1f}s] {indent}{message}")

    def load_task_data(self, task_num: int) -> dict:
        with open(self.data_dir / f"task{task_num:03d}.json", 'r') as f:
            return json.load(f)

    def read_solution(self, task_num: int) -> bytes:
        task_file = self.baseline_dir / f"task{task_num:03d}.py"
        if not task_file.exists():
            return None
        with open(task_file, 'rb') as f:
            return f.read()

    def verify(self, code_bytes: bytes, task_num: int, quick=True) -> bool:
        """Verify solution correctness"""
        try:
            code = code_bytes.decode('utf-8', errors='ignore')

            # Fix common compressed pattern: bytes("...") missing encoding
            if 'import zlib' in code and 'bytes(' in code:
                code = re.sub(r'bytes\((""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')\)', r'bytes(\1,"latin1")', code, flags=re.S)

            task_data = self.load_task_data(task_num)

            namespace = {}
            exec(code, namespace)
            if 'p' not in namespace:
                return False

            program = namespace['p']

            # Test examples
            examples = task_data['train'] + task_data['test']
            if not quick:
                examples += task_data.get('arc-gen', [])[:10]

            for example in examples:
                input_copy = copy.deepcopy(example['input'])
                result = program(input_copy)
                if json.dumps(result) != json.dumps(example['output']):
                    return False

            return True
        except:
            return False

    def is_compressed_stub(self, code_str: str) -> bool:
        """Check if this is a zlib-compressed stub"""
        return 'import zlib' in code_str and 'exec(zlib.decompress(' in code_str and 'bytes(' in code_str

    def try_decompress_and_optimize(self, code_str: str, task_num: int) -> list:
        """For zlib stubs, try to find shorter algorithmic alternatives"""
        variants = []
        
        if not self.is_compressed_stub(code_str):
            return variants
        
        try:

            if 'import zlib' in code_str:
                simplified = code_str.replace('import zlib\n', '')
                if simplified != code_str:
                    variants.append(simplified)
            
            # Try shorter function definitions
            if 'def p(g):' in code_str:
                variants.append(code_str.replace('def p(g):', 'p=lambda g:'))
        except:
            pass
        
        return variants

    def _minimize_indentation(self, code: str) -> str:
        """Minimize indentation to single spaces"""
        lines = []
        for line in code.split('\n'):
            if not line.strip():
                continue
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 0:
                level = 1 if indent < 4 else ((indent + 3) // 4)
                lines.append(' ' * level + stripped.rstrip())
            else:
                lines.append(stripped.rstrip())
        return '\n'.join(lines)

    def generate__variants(self, code: str, task_num: int) -> list:
        """Generate hundreds of optimization variants (from _optimizer)"""
        variants = set([code])

        self.log(f"Task {task_num:03d}: Generating optimization variants...", 1)

        # === PHASE 1: Basic Optimizations ===
        basic_transforms = [
            (lambda c: re.sub(r',\s+', ',', c), "comma spaces"),
            (lambda c: re.sub(r':\s+', ':', c), "colon spaces"),
            (lambda c: re.sub(r'(\w)\s*=\s*([^=])', r'\1=\2', c), "equals spaces"),
            (lambda c: c.replace('return [', 'return['), "return spacing"),
            (lambda c: c.replace('return [[', 'return[['), "return double bracket"),
            (lambda c: c.replace('return (', 'return('), "return paren"),
            (lambda c: '\n'.join(l.rstrip() for l in c.split('\n') if l.strip()), "trailing whitespace"),
            (lambda c: re.sub(r'\bor\s+(\w+)\s+for\b', r'|\1 for', c), "or to | in comprehension"),
            (lambda c: re.sub(r'\bor\s+(\w+)\s*\]', r'|\1]', c), "or to | before ]"),
            (lambda c: re.sub(r'\bor\s+(\w+)\s*\)', r'|\1)', c), "or to | before )"),
            (lambda c: re.sub(r'\bor\s+(\w)', r'|\1', c), "or to | general"),
            (lambda c: re.sub(r'if\s+(\w+)\s*==\s*0\s*:', r'if not \1:', c), "==0 to not"),
            (lambda c: re.sub(r'if\s+(\w+)\s*!=\s*0\s*:', r'if \1:', c), "!=0 removal"),
            (lambda c: re.sub(r'if\s+(\w+)\s*==\s*True\s*:', r'if \1:', c), "==True removal"),
            (lambda c: re.sub(r'if\s+(\w+)\s*==\s*False\s*:', r'if not \1:', c), "==False to not"),
            (lambda c: re.sub(r'(\w+)\s*==\s*0', r'not \1', c), "==0 in expressions"),
            (lambda c: re.sub(r'(\w+)\s*!=\s*0', r'\1', c), "!=0 in expressions"),
            (lambda c: self._minimize_indentation(c), "indent minimization"),
        ]

        for transform, desc in basic_transforms:
            try:
                result = transform(code)
                if result != code and result:
                    variants.add(result)
            except:
                pass

        self.log(f"After basic transforms: {len(variants)} variants", 2)

        # === PHASE 2: Combinations ===
        base_variants = list(variants)[:30]
        for base in base_variants:
            for t1, t2 in combinations(basic_transforms[:10], 2):
                try:
                    result = t1[0](base)
                    result = t2[0](result)
                    if result != base and result:
                        variants.add(result)
                except:
                    pass

        self.log(f"After combinations: {len(variants)} variants", 2)

        # === PHASE 3: Aggressive Space Removal ===
        aggressive_transforms = [
            (lambda c: re.sub(r'\s*([+\-*/%<>=!&|])\s*', r'\1', c), "aggressive operators"),
            (lambda c: re.sub(r'\s+', ' ', c), "multiple spaces to single"),
            (lambda c: re.sub(r'\s*;\s*', ';', c), "semicolon spaces"),
            (lambda c: re.sub(r'\s*,\s*', ',', c), "comma spaces aggressive"),
            (lambda c: re.sub(r'\s*:\s*', ':', c), "colon spaces aggressive"),
            (lambda c: re.sub(r'\s*\(\s*', '(', c), "paren spaces"),
            (lambda c: re.sub(r'\s*\)\s*', ')', c), "closing paren spaces"),
            (lambda c: re.sub(r'\s*\[\s*', '[', c), "bracket spaces"),
            (lambda c: re.sub(r'\s*\]\s*', ']', c), "closing bracket spaces"),
        ]

        for base in list(variants)[:60]:
            for transform, desc in aggressive_transforms:
                try:
                    result = transform(base)
                    if result != base and result:
                        variants.add(result)
                except:
                    pass

        self.log(f"After aggressive transforms: {len(variants)} variants", 2)

        # === PHASE 4: Pattern-Specific Optimizations ===
        # Community golf tricks from simplification-is-key
        
        # 1. Lambda conversion (saves 4+ bytes)
        if 'def p(g):' in code and 'return' in code:
            # Try converting def to lambda
            lambda_version = code.replace('def p(g):', 'p=lambda g:').replace('\n return', '')
            if len(lambda_version) < len(code):
                variants.add(lambda_version)

        # 2. Flattening grids with sum(g,[])
        if 'for y in range' in code and 'for x in range' in code:
            flat_version = re.sub(
                r'for\s+(\w+)\s+in\s+range\(len\(g\)\):\s*for\s+(\w+)\s+in\s+range\(len\(g\[0\]\)\):',
                r'for i,v in enumerate(sum(g,[])):',
                code
            )
            if flat_version != code:
                variants.add(flat_version)

        # 3. Use zip(*g) for column processing
        if 'for x in range(len(g[0]))' in code:
            col_version = code.replace('for x in range(len(g[0])):', 'for j,c in enumerate(zip(*g)):')
            if col_version != code:
                variants.add(col_version)

        # 4. Set operations for finding unique values
        if 'max(' in code or 'min(' in code:
            set_version = re.sub(r'max\(([^)]+)\)', r'max(set(\1))', code)
            set_version = re.sub(r'min\(([^)]+)\)', r'min(set(\1))', set_version)
            if set_version != code:
                variants.add(set_version)

        # 5. Walrus operator for assignment in expressions
        if '=' in code and 'if' in code:
            walrus_version = re.sub(r'(\w+)\s*=\s*([^;]+)\s*\n\s*if\s+\1', r'if(\1:=\2)', code)
            if walrus_version != code:
                variants.add(walrus_version)

        # 6. Bit tricks: ~(x-y) instead of y-x-1
        if '-1' in code:
            bit_version = re.sub(r'(\w+)\s*-\s*(\w+)\s*-\s*1', r'~(\1-\2)', code)
            if bit_version != code:
                variants.add(bit_version)

        # 7. In-place list operations
        if '[:]' in code:
            inplace_version = code.replace('result=[row[:] for row in g]', 'result=g')
            if inplace_version != code:
                variants.add(inplace_version)

        # 8. Slice assignment instead of loops
        if 'for' in code and '=' in code:
            slice_version = re.sub(
                r'for\s+(\w+)\s+in\s+range\((\w+),(\w+)\):\s*(\w+)\[(\w+)\]\s*=\s*([^\n]+)',
                r'\4[\2:\3]=[\6]*(\3-\2)',
                code
            )
            if slice_version != code:
                variants.add(slice_version)

        # 9. Dictionary setdefault for caching
        if 'if' in code and 'in' in code and '{}' in code:
            setdefault_version = re.sub(
                r'if\s+(\w+)\s+not\s+in\s+(\w+):\s*\2\[\1\]\s*=\s*([^\n]+)',
                r'\2.setdefault(\1,\3)',
                code
            )
            if setdefault_version != code:
                variants.add(setdefault_version)

        # 10. String eval for rotation (from community-baselines)
        if 'zip(*' in code and 'reverse' in code:
            eval_version = code.replace('zip(*g[::-1])', 'eval(str(g).replace(...))') 
            if eval_version != code and len(eval_version) < len(code):
                variants.add(eval_version)

        # Range aliasing
        if code.count('range') >= 2:
            if 'def p(' in code:
                v = re.sub(r'def p\(([^)]*)\):', r'def p(\1,R=range):', code)
                v = v.replace('(,R=range)', '(R=range)')
                v = v.replace('range(', 'R(')
                variants.add(v)
            elif 'p=lambda' in code:
                v = code.replace('p=lambda ', 'p=lambda R=range,')
                v = v.replace('range(', 'R(')
                variants.add(v)

        # Len aliasing
        if code.count('len(') >= 2:
            if 'def p(' in code:
                v = re.sub(r'def p\(([^)]*)\):', r'def p(\1,L=len):', code)
                v = v.replace('(,L=len)', '(L=len)')
                v = v.replace('len(', 'L(')
                variants.add(v)

        # Enumerate aliasing
        if code.count('enumerate(') >= 2:
            if 'def p(' in code:
                v = re.sub(r'def p\(([^)]*)\):', r'def p(\1,E=enumerate):', code)
                v = v.replace('(,E=enumerate)', '(E=enumerate)')
                v = v.replace('enumerate(', 'E(')
                variants.add(v)

        # === PHASE 6: Ultra-Aggressive Optimizations ===
        # Advanced golf tricks from expert community

        # 1. Extreme variable name reduction
        if any(var in code for var in ['result', 'temp', 'count', 'index']):
            ultra_version = code
            ultra_version = re.sub(r'\bresult\b', 'r', ultra_version)
            ultra_version = re.sub(r'\btemp\b', 't', ultra_version)
            ultra_version = re.sub(r'\bcount\b', 'c', ultra_version)
            ultra_version = re.sub(r'\bindex\b', 'i', ultra_version)
            if ultra_version != code:
                variants.add(ultra_version)

        # 2. Function parameter aliasing (ultra-compact)
        if 'def p(g,' in code:
            param_version = re.sub(r'def p\(g,([^)]*)\):', r'def p(g,\1,R=range,L=len,E=enumerate):', code)
            param_version = param_version.replace('range(', 'R(').replace('len(', 'L(').replace('enumerate(', 'E(')
            if param_version != code and len(param_version) < len(code):
                variants.add(param_version)

        # 3. List comprehension extreme golfing
        if 'for' in code and '[' in code and ']' in code:
            comp_version = re.sub(r'\[([^\]]+)\s+for\s+(\w+)\s+in\s+range\(([^)]+)\)\s+for\s+(\w+)\s+in\s+range\(([^)]+)\)\]',
                                 r'[\1 for\2 in R(\3)for\4 in R(\5)]', code)
            if comp_version != code:
                variants.add(comp_version)

        # 4. String manipulation for grid operations
        if 'str(' in code or 'replace(' in code:
            str_version = re.sub(r'str\(([^)]+)\)\.replace\(([^,]+),\s*([^)]+)\)',
                               r'\1.replace(\2,\3)', code)
            if str_version != code:
                variants.add(str_version)

        # 5. Extreme operator compaction
        if ' ' in code:
            op_version = re.sub(r'\s*([+\-*/%=<>!&|^~])\s*', r'\1', code)
            op_version = re.sub(r'\s*([()[\]{}.,:])\s*', r'\1', op_version)
            if op_version != code and len(op_version) < len(code):
                variants.add(op_version)

        # 6. Function inlining for small functions
        if 'def ' in code and len(code.split('def ')) <= 3:
            # Try inlining small helper functions
            inline_version = re.sub(r'def\s+(\w+)\([^)]*\):\s*return\s+([^;]+);?\s*\n', r'', code, flags=re.MULTILINE)
            if inline_version != code:
                variants.add(inline_version)

        # 7. Magic number optimization
        if any(str(n) in code for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            magic_version = re.sub(r'\b([2-9]|10)\b', lambda m: str(int(m.group(1)) - 1) if int(m.group(1)) > 1 else str(int(m.group(1)) + 1), code)
            if magic_version != code:
                variants.add(magic_version)

        # 8. List slicing optimization
        if '[:' in code:
            slice_version = re.sub(r'(\w+)\[:(\d+)\]', r'\1[:\2]', code)  # Remove unnecessary slice start
            slice_version = re.sub(r'(\w+)\[(\d+):\]', r'\1[\2:]', slice_version)  # Remove unnecessary slice end
            if slice_version != code:
                variants.add(slice_version)

        # 9. Conditional expression golfing
        if 'if ' in code and 'else' in code:
            cond_version = re.sub(r'(\w+)\s*=\s*([^;]+?)\s+if\s+([^;]+?)\s+else\s+([^;]+)',
                                 r'\1=\2 if\3 else\4', code)
            if cond_version != code:
                variants.add(cond_version)

        # 10. Generator expression conversion
        if 'sum(' in code or 'max(' in code or 'min(' in code:
            gen_version = re.sub(r'\[([^\]]+)\s+for\s+(\w+)\s+in\s+([^]]+)\]',
                               r'(\1 for\2 in\3)', code)
            if gen_version != code:
                variants.add(gen_version)

        # 11. Import removal (if not needed)
        if 'import ' in code:
            importless_version = re.sub(r'import\s+\w+\s*\n', '', code)
            if importless_version != code and len(importless_version) < len(code):
                variants.add(importless_version)

        # 12. String literal optimization
        if '"""' in code or "'''" in code:
            str_version = code.replace('"""', '"').replace("'''", "'")
            if str_version != code:
                variants.add(str_version)

        # 13. Boolean expression golfing
        if 'True' in code or 'False' in code:
            bool_version = code.replace('True', '1').replace('False', '0')
            if bool_version != code:
                variants.add(bool_version)

        # 14. Loop unrolling for small ranges
        if 'range(2)' in code or 'range(3)' in code:
            unroll_version = re.sub(r'for\s+(\w+)\s+in\s+range\(2\):([^;]+);?',
                                   r'\1=0;\2;\1=1;\2', code)
            if unroll_version != code:
                variants.add(unroll_version)

        # === PHASE 8: Extreme Golf Tricks ===

        # 16. Extreme list comprehension golfing
        if 'for' in code and '[' in code and ']' in code:
            extreme_comp = re.sub(r'\[([^\]]+)\s+for\s+(\w+)\s+in\s+range\(([^)]+)\)\s+for\s+(\w+)\s+in\s+range\(([^)]+)\)\]',
                                 r'[\1 for\2 in R(\3)for\4 in R(\5)]', code)
            extreme_comp = re.sub(r'for\s+(\w+)\s+in\s+range\(([^)]+)\)', r'for\1 in R(\2)', extreme_comp)
            if extreme_comp != code:
                variants.add(extreme_comp)

        # 17. Ultra-compact function definitions
        if 'def p(' in code:
            ultra_def = re.sub(r'def p\(([^)]*)\):', r'def p(\1,R=range,L=len,E=enumerate,Z=zip,S=sum,M=max,N=min):', code)
            ultra_def = ultra_def.replace('range(', 'R(').replace('len(', 'L(').replace('enumerate(', 'E(')
            ultra_def = ultra_def.replace('zip(', 'Z(').replace('sum(', 'S(').replace('max(', 'M(').replace('min(', 'N(')
            if ultra_def != code and len(ultra_def) < len(code):
                variants.add(ultra_def)

        # 18. Advanced string manipulation
        if 'str(' in code:
            str_manip = re.sub(r'str\(([^)]+)\)', r'str(\1)', code)  # Remove unnecessary parentheses
            if str_manip != code:
                variants.add(str_manip)

        # 19. Extreme operator chaining
        if '==' in code or '!=' in code:
            chain_version = re.sub(r'(\w+)\s*==\s*(\w+)', r'\1==\2', code)
            chain_version = re.sub(r'(\w+)\s*!=\s*(\w+)', r'\1!=\2', chain_version)
            if chain_version != code:
                variants.add(chain_version)

        # 20. List method golfing
        if '.append(' in code or '.extend(' in code:
            list_golf = re.sub(r'(\w+)\.append\(([^)]+)\)', r'\1+=[[\2]]', code)
            if list_golf != code:
                variants.add(list_golf)

        # 21. Dictionary key golfing
        if '{' in code and ':' in code:
            dict_golf = re.sub(r'\{([^}]+)\}', r'{\1}', code)  # Remove unnecessary spaces
            if dict_golf != code:
                variants.add(dict_golf)

        # 22. Function call golfing
        if '(' in code and ')' in code:
            call_golf = re.sub(r'(\w+)\s*\(\s*([^)]*)\s*\)', r'\1(\2)', code)
            if call_golf != code:
                variants.add(call_golf)

        # 23. Variable name extreme golfing
        if any(var in code for var in ['input', 'output', 'result', 'grid', 'data']):
            var_golf = code
            var_golf = re.sub(r'\binput\b', 'i', var_golf)
            var_golf = re.sub(r'\boutput\b', 'o', var_golf)
            var_golf = re.sub(r'\bresult\b', 'r', var_golf)
            var_golf = re.sub(r'\bgrid\b', 'g', var_golf)
            var_golf = re.sub(r'\bdata\b', 'd', var_golf)
            if var_golf != code:
                variants.add(var_golf)

        # 24. Extreme indentation removal
        if '\n  ' in code:
            indent_golf = re.sub(r'\n\s+', '\n', code)
            if indent_golf != code:
                variants.add(indent_golf)

        # 25. Comment removal (if any)
        if '#' in code:
            no_comment = re.sub(r'#.*', '', code)
            if no_comment != code:
                variants.add(no_comment)

        # 26. Whitespace extreme removal
        if '  ' in code:
            space_golf = re.sub(r'\s+', ' ', code)
            if space_golf != code:
                variants.add(space_golf)

        # 27. Single character variable names
        if len(code) > 100:
            single_var = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', lambda m: m.group(1)[0] if len(m.group(1)) > 1 else m.group(1), code)
            if single_var != code:
                variants.add(single_var)

        # 28. Extreme list comprehension
        if 'for' in code and 'if' in code:
            extreme_list = re.sub(r'\[([^\]]+)\s+for\s+(\w+)\s+in\s+([^]]+)\s+if\s+([^\]]+)\]',
                                 r'[\1 for\2 in\3 if\4]', code)
            if extreme_list != code:
                variants.add(extreme_list)

        # 29. Tuple unpacking golfing
        if ',' in code and '=' in code:
            tuple_golf = re.sub(r'(\w+),\s*(\w+)\s*=\s*([^;]+)', r'\1,\2=\3', code)
            if tuple_golf != code:
                variants.add(tuple_golf)

        # 30. Advanced regex for pattern matching
        if 're.' in code:
            regex_golf = re.sub(r're\.search\(([^,]+),\s*([^)]+)\)', r're.search(\1,\2)', code)
            if regex_golf != code:
                variants.add(regex_golf)

        self.log(f"After extreme golf tricks: {len(variants)} variants", 2)

        # === PHASE 9: Enhanced C++ Ultra-Fast Optimization ===
        # Integrate enhanced C++ optimizations for ultra-fast processing

        # try:
        #     from cpp_optimizer_wrapper import CppOptimizerWrapper
        #     cpp_wrapper = CppOptimizerWrapper()

        #     # Apply C++ optimizations to all variants
        #     cpp_variants = []
        #     for variant in list(variants):
        #         cpp_optimized = cpp_wrapper.optimize_with_cpp(variant, task_num)
        #         if cpp_optimized != variant and len(cpp_optimized) < len(variant):
        #             cpp_variants.append(cpp_optimized)

        #     variants.update(cpp_variants)
        #     self.log(f"After enhanced C++ optimizations: {len(variants)} variants", 2)

        # except ImportError:
        #     self.log("Enhanced C++ optimizer not available, skipping C++ optimizations", 2)
        # except Exception as e:
        #     self.log(f"Enhanced C++ optimization error: {e}, continuing without C++", 2)

        # === PHASE 10:  Combinations ===
        sorted_variants = sorted(list(variants), key=len)[:50]  # Increased from 40

        for base in sorted_variants:
            ultra = base
            for transform, _ in basic_transforms:
                try:
                    ultra = transform(ultra)
                except:
                    pass
            if ultra != base and ultra:
                variants.add(ultra)

            # Apply aggressive transforms again
            for transform, _ in aggressive_transforms:
                try:
                    result = transform(ultra)
                    if result and result != ultra:
                        variants.add(result)
                except:
                    pass

        self.log(f" variant count after ultra combinations: {len(variants)} variants", 2)

        return list(variants)

    def optimize_task__final(self, task_num: int) -> tuple:
        """  optimization of single task"""
        original_bytes = self.read_solution(task_num)
        if not original_bytes:
            return 0, 0

        original_size = len(original_bytes)

        # Verify original
        if not self.verify(original_bytes, task_num):
            output_file = self.output_dir / f"task{task_num:03d}.py"
            with open(output_file, 'wb') as f:
                f.write(original_bytes)
            return original_size, original_size

        # Decode
        try:
            original_str = original_bytes.decode('utf-8')
        except:
            try:
                original_str = original_bytes.decode('latin-1')
            except:
                output_file = self.output_dir / f"task{task_num:03d}.py"
                with open(output_file, 'wb') as f:
                    f.write(original_bytes)
                return original_size, original_size

        # Generate variants
        variants = self.generate__variants(original_str, task_num)

        # Test variants (sort by size first)
        self.log(f"Task {task_num:03d}: Testing {len(variants)} variants...", 1)
        variants_by_size = sorted(variants, key=len)

        best_bytes = original_bytes
        best_size = original_size
        tested = 0
        improvements_found = 0

        for i, variant in enumerate(variants_by_size):
            if i > 0 and i % 100 == 0:
                self.log(f"Progress: {i}/{len(variants)} tested, {improvements_found} improvements found", 2)

            try:
                variant_bytes = variant.encode('utf-8')
                if len(variant_bytes) < best_size:
                    tested += 1
                    if self.verify(variant_bytes, task_num, quick=True):
                        if self.verify(variant_bytes, task_num, quick=False):
                            improvement = best_size - len(variant_bytes)
                            improvements_found += 1
                            self.log(f"âœ“ Improvement: {best_size} â†’ {len(variant_bytes)} (-{improvement})", 2)
                            best_bytes = variant_bytes
                            best_size = len(variant_bytes)
            except:
                continue

        self.log(f"Task {task_num:03d}: Tested {tested} candidates, found {improvements_found} improvements", 1)

        # Save best
        output_file = self.output_dir / f"task{task_num:03d}.py"
        with open(output_file, 'wb') as f:
            f.write(best_bytes)

        if best_size < original_size:
            saved = original_size - best_size
            self.improvements.append((task_num, saved, original_size, best_size))
            self.log(f"âœ… Task {task_num:03d} IMPROVED: {original_size} â†’ {best_size} (-{saved} bytes)", 0)

        return original_size, best_size

    def run(self):
        """Main execution - prioritize longest tasks"""
        print("="*70)
        print("  OPTIMIZER - Maximum Effort with Compressed File Handling")
        print("="*70)
        self.log("Analyzing all tasks to prioritize longest ones...")

        # First pass: get all task sizes
        task_list = []
        for task_num in range(1, 401):
            code_bytes = self.read_solution(task_num)
            if code_bytes:
                size = len(code_bytes)
                task_list.append((task_num, size))

        # Sort by size (largest first)
        task_list.sort(key=lambda x: -x[1])

        self.log(f"Found {len(task_list)} tasks")
        self.log(f"Largest task: {task_list[0][1]} bytes (Task {task_list[0][0]})")
        self.log(f"Smallest task: {task_list[-1][1]} bytes (Task {task_list[-1][0]})")
        self.log("")
        self.log("Optimizing in order of size (largest first for max impact)...")
        print("="*70)

        total_orig = 0
        total_final = 0

        for idx, (task_num, orig_size) in enumerate(task_list):
            print()
            self.log(f"=== Task {task_num:03d} ({idx+1}/400) - {orig_size} bytes ===", 0)

            orig, final = self.optimize_task__final(task_num)
            total_orig += orig
            total_final += final

            # Progress summary every 50 tasks
            if (idx + 1) % 50 == 0:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                print()
                print("="*70)
                self.log(f"PROGRESS: {idx+1}/400 tasks ({(idx+1)/4:.0f}%)")
                self.log(f"Improvements: {len(self.improvements)} tasks")
                self.log(f"Bytes saved: {total_orig - total_final}")
                self.log(f"Time: {elapsed/60:.1f} minutes")
                est_remaining = (elapsed / (idx + 1)) * (400 - idx - 1)
                self.log(f"Estimated remaining: {est_remaining/60:.1f} minutes")
                print("="*70)

        #  summary
        total_saved = total_orig - total_final
        elapsed = (datetime.now() - self.start_time).total_seconds()

        print()
        print("="*70)
        print("  OPTIMIZATION COMPLETE!")
        print("="*70)
        self.log(f"Total time: {elapsed/60:.1f} minutes")
        self.log(f"Tasks improved: {len(self.improvements)}/400 ({len(self.improvements)/4:.1f}%)")
        self.log(f"Total bytes saved: {total_saved}")
        print("="*70)

        if self.improvements:
            print()
            print("ALL IMPROVEMENTS:")
            print("-"*70)
            for task_num, saved, orig, final in sorted(self.improvements, key=lambda x: -x[1]):
                print(f"Task {task_num:03d}: {orig} â†’ {final} bytes (-{saved})")

        # Create submission
        self.log("Creating submission.zip...")
        zip_path = Path('submission.zip')
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for task_num in range(1, 401):
                task_file = self.output_dir / f"task{task_num:03d}.py"
                if task_file.exists():
                    zipf.write(task_file, f"task{task_num:03d}.py")

        zip_size = zip_path.stat().st_size

        # Calculate score
        score = sum(max(1, 2500 - (self.output_dir / f"task{task_num:03d}.py").stat().st_size)
                   for task_num in range(1, 401)
                   if (self.output_dir / f"task{task_num:03d}.py").exists())

        print()
        print("="*70)
        print("SUBMISSION READY")
        print("="*70)
        print(f"ğŸ“� File: submission.zip ({zip_size/1024:.1f} KB)")
        print(f"ğŸ�¯ Score: {score:,.0f} points (+{total_saved})")
        print(f"â�±ï¸�  Time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"âœ… Improvements: {len(self.improvements)}/400 tasks")
        print("="*70)

        # Save report
        self._save_report(total_orig, total_final, score, elapsed)

    def _save_report(self, total_orig, total_final, score, elapsed):
        """Save detailed report"""
        report_path = Path('__OPTIMIZATION_REPORT.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("#   Optimization Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Optimization Time**: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)\n")
            f.write(f"**Method**: Maximum effort with compressed file handling\n\n")

            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Tasks Optimized | 400/400 |\n")
            f.write(f"| Tasks Improved | {len(self.improvements)} ({len(self.improvements)/4:.1f}%) |\n")
            f.write(f"| Total Saved | {total_orig - total_final} bytes |\n")
            f.write(f"| Final Score | {score:,.0f} points |\n")
            f.write(f"| Points Gained | +{total_orig - total_final} |\n\n")

            if self.improvements:
                f.write("## Improvements by Task\n\n")
                for task_num, saved, orig, final in sorted(self.improvements, key=lambda x: -x[1]):
                    f.write(f"- **Task {task_num:03d}**: {orig} -> {final} bytes (-{saved})\n")

        self.log(f"Saved report to {report_path}")


def main():
    optimizer = Optimizer()
    optimizer.run()


if __name__ == '__main__':
    main()


