# ============================================================
# CELL 1: Install Required Libraries
# ============================================================

!pip install -q google-adk google-generativeai pylint bandit radon astroid rich

print("âœ… All libraries installed successfully!")


# ============================================================
# CELL 2: Import Libraries & Configure Environment
# ============================================================

import sys
import os
import ast
import json
import time
import re
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Import ADK components
try:
    from google.genai import types
    from google.adk.agents import LlmAgent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import InMemoryRunner
    from google.adk.sessions import InMemorySessionService
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("âš ï¸� Google ADK not available. Running in limited mode.")

# Import Gemini AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("âš ï¸� Google Generative AI not available.")

# Configure API Key
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    API_CONFIGURED = True
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    API_CONFIGURED = bool(GOOGLE_API_KEY)
    if not API_CONFIGURED:
        print("âš ï¸� GOOGLE_API_KEY not found. AI features will be limited.")

if GENAI_AVAILABLE and API_CONFIGURED:
    genai.configure(api_key=GOOGLE_API_KEY)

print("âœ… Libraries imported successfully!")
print(f"ğŸ“¦ ADK Available: {ADK_AVAILABLE}")
print(f"ğŸ¤– Gemini AI Available: {GENAI_AVAILABLE and API_CONFIGURED}")


# ============================================================
# CELL 3: Enums and Constants
# ============================================================

class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class IssueCategory(Enum):
    """Categories of code issues"""
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    COMPLEXITY = "complexity"
    DOCUMENTATION = "documentation"
    SYNTAX = "syntax"
    BEST_PRACTICE = "best_practice"
    MAINTAINABILITY = "maintainability"

class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    UNKNOWN = "unknown"

# Severity colors for display
SEVERITY_COLORS = {
    Severity.CRITICAL: "ğŸ”´",
    Severity.HIGH: "ğŸŸ ",
    Severity.MEDIUM: "ğŸŸ¡",
    Severity.LOW: "ğŸ”µ",
    Severity.INFO: "âšª"
}

# Security patterns to detect
SECURITY_PATTERNS = {
    "python": {
        r"eval\s*\(": ("Use of eval()", Severity.CRITICAL),
        r"exec\s*\(": ("Use of exec()", Severity.CRITICAL),
        r"os\.system\s*\(": ("Use of os.system()", Severity.HIGH),
        r"subprocess\.call\s*\(.*shell\s*=\s*True": ("Shell injection risk", Severity.CRITICAL),
        r"pickle\.loads?": ("Unsafe deserialization with pickle", Severity.HIGH),
        r"yaml\.load\s*\([^)]*\)(?!.*Loader)": ("Unsafe YAML loading", Severity.HIGH),
        r"__import__\s*\(": ("Dynamic import", Severity.MEDIUM),
        r"input\s*\(": ("User input without validation", Severity.LOW),
        r"password\s*=\s*[\"'][^\"']+[\"']": ("Hardcoded password", Severity.CRITICAL),
        r"api_key\s*=\s*[\"'][^\"']+[\"']": ("Hardcoded API key", Severity.CRITICAL),
        r"secret\s*=\s*[\"'][^\"']+[\"']": ("Hardcoded secret", Severity.CRITICAL),
    },
    "javascript": {
        r"eval\s*\(": ("Use of eval()", Severity.CRITICAL),
        r"innerHTML\s*=": ("XSS vulnerability via innerHTML", Severity.HIGH),
        r"document\.write\s*\(": ("Use of document.write()", Severity.MEDIUM),
        r"\$\(.*\)\.html\s*\(": ("Potential XSS in jQuery", Severity.HIGH),
    }
}

print("âœ… Constants and enums defined!")


# ============================================================
# CELL 4: Data Classes for Structured Results
# ============================================================

@dataclass
class CodeIssue:
    """Represents a single code issue"""
    message: str
    severity: Severity
    category: IssueCategory
    line_number: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "line_number": self.line_number,
            "column": self.column,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet
        }
    
    def __str__(self) -> str:
        loc = f"Line {self.line_number}" if self.line_number else "General"
        return f"{SEVERITY_COLORS[self.severity]} [{self.severity.value.upper()}] {loc}: {self.message}"

@dataclass
class CodeMetrics:
    """Code quality metrics"""
    lines_of_code: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    maintainability_index: float = 100.0
    function_count: int = 0
    class_count: int = 0
    avg_function_length: float = 0.0
    max_function_length: int = 0
    nesting_depth: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "lines_of_code": self.lines_of_code,
            "blank_lines": self.blank_lines,
            "comment_lines": self.comment_lines,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "maintainability_index": round(self.maintainability_index, 2),
            "function_count": self.function_count,
            "class_count": self.class_count,
            "avg_function_length": round(self.avg_function_length, 2),
            "max_function_length": self.max_function_length,
            "nesting_depth": self.nesting_depth
        }
    
    def get_grade(self) -> str:
        """Get letter grade based on maintainability index"""
        if self.maintainability_index >= 80:
            return "A"
        elif self.maintainability_index >= 60:
            return "B"
        elif self.maintainability_index >= 40:
            return "C"
        elif self.maintainability_index >= 20:
            return "D"
        else:
            return "F"

@dataclass
class ReviewResult:
    """Complete code review result"""
    code_hash: str
    language: Language
    timestamp: str
    issues: List[CodeIssue] = field(default_factory=list)
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    ai_suggestions: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "code_hash": self.code_hash,
            "language": self.language.value,
            "timestamp": self.timestamp,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics.to_dict(),
            "ai_suggestions": self.ai_suggestions,
            "execution_time": round(self.execution_time, 3),
            "summary": self.get_summary()
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        severity_counts = {s.value: 0 for s in Severity}
        category_counts = {c.value: 0 for c in IssueCategory}
        
        for issue in self.issues:
            severity_counts[issue.severity.value] += 1
            category_counts[issue.category.value] += 1
        
        return {
            "total_issues": len(self.issues),
            "by_severity": severity_counts,
            "by_category": category_counts,
            "quality_grade": self.metrics.get_grade(),
            "risk_level": self._calculate_risk_level()
        }
    
    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level"""
        critical_count = sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
        high_count = sum(1 for i in self.issues if i.severity == Severity.HIGH)
        
        if critical_count > 0:
            return "CRITICAL"
        elif high_count > 2:
            return "HIGH"
        elif high_count > 0:
            return "MEDIUM"
        else:
            return "LOW"

print("âœ… Data classes defined!")


# ============================================================
# CELL 5: Observability System - Enhanced Logging & Metrics
# ============================================================

class ObservabilityLogger:
    """Enhanced observability system for logging, tracing, and metrics"""
    
    def __init__(self, verbose: bool = True):
        self.logs: List[Dict] = []
        self.metrics: Dict[str, Any] = {
            'total_reviews': 0,
            'total_issues_found': 0,
            'issues_by_severity': {s.value: 0 for s in Severity},
            'issues_by_category': {c.value: 0 for c in IssueCategory},
            'avg_execution_time': 0.0,
            'total_execution_time': 0.0,
            'languages_analyzed': {}
        }
        self.verbose = verbose
        self._start_times: Dict[str, float] = {}
    
    def log_action(self, agent_name: str, action: str, details: Any = None):
        """Log an agent action"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'action': action,
            'details': details or {}
        }
        self.logs.append(log_entry)
        if self.verbose:
            print(f"[{agent_name}] {action}")
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self._start_times[operation] = time.time()
    
    def stop_timer(self, operation: str) -> float:
        """Stop timing and return elapsed time"""
        if operation in self._start_times:
            elapsed = time.time() - self._start_times[operation]
            del self._start_times[operation]
            return elapsed
        return 0.0
    
    def update_metrics(self, result: ReviewResult):
        """Update metrics from a review result"""
        self.metrics['total_reviews'] += 1
        self.metrics['total_issues_found'] += len(result.issues)
        self.metrics['total_execution_time'] += result.execution_time
        self.metrics['avg_execution_time'] = (
            self.metrics['total_execution_time'] / self.metrics['total_reviews']
        )
        
        # Update language stats
        lang = result.language.value
        if lang not in self.metrics['languages_analyzed']:
            self.metrics['languages_analyzed'][lang] = 0
        self.metrics['languages_analyzed'][lang] += 1
        
        # Update severity and category counts
        for issue in result.issues:
            self.metrics['issues_by_severity'][issue.severity.value] += 1
            self.metrics['issues_by_category'][issue.category.value] += 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.copy()
    
    def get_logs(self, limit: int = None) -> List[Dict]:
        """Get logs, optionally limited"""
        if limit:
            return self.logs[-limit:]
        return self.logs.copy()
    
    def print_metrics_summary(self):
        """Print a formatted metrics summary"""
        print("\n" + "="*60)
        print("ğŸ“Š METRICS SUMMARY")
        print("="*60)
        print(f"Total Reviews: {self.metrics['total_reviews']}")
        print(f"Total Issues Found: {self.metrics['total_issues_found']}")
        print(f"Average Execution Time: {self.metrics['avg_execution_time']:.3f}s")
        print("\nIssues by Severity:")
        for sev, count in self.metrics['issues_by_severity'].items():
            if count > 0:
                print(f"  {sev}: {count}")
        print("\nLanguages Analyzed:")
        for lang, count in self.metrics['languages_analyzed'].items():
            print(f"  {lang}: {count}")
        print("="*60)

# Initialize global logger
logger = ObservabilityLogger(verbose=True)
print("âœ… Observability system initialized!")


# ============================================================
# CELL 6: Memory System - Learning from Reviews
# ============================================================

@dataclass
class ReviewMemory:
    """Long-term memory system for learning from reviews"""
    patterns: List[Dict] = field(default_factory=list)
    preferences: Dict = field(default_factory=dict)
    common_issues: Dict[str, int] = field(default_factory=dict)
    fix_history: List[Dict] = field(default_factory=list)
    
    def store_review(self, result: ReviewResult):
        """Store a review result for future reference"""
        self.patterns.append({
            'code_hash': result.code_hash,
            'language': result.language.value,
            'issue_count': len(result.issues),
            'issues': [i.message for i in result.issues],
            'metrics': result.metrics.to_dict(),
            'timestamp': result.timestamp
        })
        
        # Track common issues
        for issue in result.issues:
            key = f"{issue.category.value}:{issue.message[:50]}"
            self.common_issues[key] = self.common_issues.get(key, 0) + 1
    
    def get_similar_reviews(self, code_hash: str) -> List[Dict]:
        """Find similar past reviews"""
        return [p for p in self.patterns if p['code_hash'] == code_hash]
    
    def get_common_issues(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most common issues"""
        sorted_issues = sorted(
            self.common_issues.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_issues[:top_n]
    
    def store_fix(self, issue: str, fix: str, success: bool):
        """Store a fix attempt for learning"""
        self.fix_history.append({
            'issue': issue,
            'fix': fix,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_preference(self, key: str, value: Any):
        """Update user preferences"""
        self.preferences[key] = value
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_reviews_stored': len(self.patterns),
            'unique_issues_tracked': len(self.common_issues),
            'fixes_recorded': len(self.fix_history),
            'preferences_set': len(self.preferences)
        }

memory = ReviewMemory()
print("âœ… Memory system initialized!")


# ============================================================
# CELL 7: Language Detection and Utilities
# ============================================================

class LanguageDetector:
    """Detect programming language from code"""
    
    PATTERNS = {
        Language.PYTHON: [
            r'^\s*def\s+\w+\s*\(',
            r'^\s*class\s+\w+',
            r'^\s*import\s+\w+',
            r'^\s*from\s+\w+\s+import',
            r':\s*$',
            r'^\s*@\w+',
        ],
        Language.JAVASCRIPT: [
            r'^\s*function\s+\w+\s*\(',
            r'^\s*const\s+\w+\s*=',
            r'^\s*let\s+\w+\s*=',
            r'^\s*var\s+\w+\s*=',
            r'=>',
            r'console\.log',
        ],
        Language.TYPESCRIPT: [
            r':\s*(string|number|boolean|any)\s*[;=)]',
            r'interface\s+\w+',
            r'type\s+\w+\s*=',
            r'<\w+>',
        ],
        Language.JAVA: [
            r'public\s+class\s+\w+',
            r'private\s+\w+\s+\w+',
            r'System\.out\.println',
            r'public\s+static\s+void\s+main',
        ],
        Language.GO: [
            r'^package\s+\w+',
            r'^func\s+\w+\s*\(',
            r'fmt\.Println',
            r':=',
        ]
    }
    
    @classmethod
    def detect(cls, code: str) -> Language:
        """Detect the programming language of the code"""
        scores = {lang: 0 for lang in Language}
        
        for lang, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    scores[lang] += 1
        
        # Get language with highest score
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 0:
            return best_lang
        return Language.UNKNOWN

class CodeUtils:
    """Utility functions for code analysis"""
    
    @staticmethod
    def compute_hash(code: str) -> str:
        """Compute a hash of the code for identification"""
        return hashlib.md5(code.encode()).hexdigest()[:12]
    
    @staticmethod
    def count_lines(code: str) -> Tuple[int, int, int]:
        """Count total, blank, and comment lines"""
        lines = code.split('\n')
        total = len(lines)
        blank = sum(1 for line in lines if not line.strip())
        comment = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))
        return total, blank, comment
    
    @staticmethod
    def extract_functions(code: str, language: Language) -> List[Dict]:
        """Extract function definitions from code"""
        functions = []
        
        if language == Language.PYTHON:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append({
                            'name': node.name,
                            'line': node.lineno,
                            'args': [arg.arg for arg in node.args.args],
                            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                        })
            except SyntaxError:
                pass
        
        return functions
    
    @staticmethod
    def extract_classes(code: str, language: Language) -> List[Dict]:
        """Extract class definitions from code"""
        classes = []
        
        if language == Language.PYTHON:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append({
                            'name': node.name,
                            'line': node.lineno,
                            'bases': [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases],
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        })
            except SyntaxError:
                pass
        
        return classes

print("âœ… Language detection and utilities ready!")


# ============================================================
# CELL 8: Code Analysis Tools - Enhanced
# ============================================================

class CodeAnalysisTools:
    """Comprehensive code analysis tools"""
    
    @staticmethod
    def analyze_ast(code: str) -> Dict:
        """Analyze code using AST"""
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            imports = []
            global_vars = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args_count': len(node.args.args),
                        'has_docstring': (isinstance(node.body[0], ast.Expr) and 
                                         isinstance(node.body[0].value, ast.Constant) and
                                         isinstance(node.body[0].value.value, str)) if node.body else False
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods_count': sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                    })
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    imports.append(f"{node.module}.{', '.join(alias.name for alias in node.names)}")
                elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.col_offset == 0:  # Global scope
                        global_vars.append(node.targets[0].id)
            
            return {
                'valid': True,
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'global_vars': global_vars
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': str(e),
                'line': e.lineno,
                'offset': e.offset
            }
    
    @staticmethod
    def calculate_complexity(code: str) -> Dict:
        """Calculate cyclomatic and cognitive complexity"""
        try:
            tree = ast.parse(code)
            
            cyclomatic = 1
            cognitive = 0
            nesting_depth = 0
            max_nesting = 0
            
            def analyze_node(node, depth=0):
                nonlocal cyclomatic, cognitive, max_nesting
                
                # Cyclomatic complexity
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    cyclomatic += 1
                    cognitive += 1 + depth  # Cognitive adds nesting penalty
                elif isinstance(node, ast.BoolOp):
                    cyclomatic += len(node.values) - 1
                    cognitive += len(node.values) - 1
                elif isinstance(node, (ast.And, ast.Or)):
                    cyclomatic += 1
                
                # Track nesting
                new_depth = depth
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    new_depth = depth + 1
                    max_nesting = max(max_nesting, new_depth)
                
                for child in ast.iter_child_nodes(node):
                    analyze_node(child, new_depth)
            
            analyze_node(tree)
            
            return {
                'cyclomatic': cyclomatic,
                'cognitive': cognitive,
                'max_nesting': max_nesting
            }
        except:
            return {'cyclomatic': 0, 'cognitive': 0, 'max_nesting': 0}
    
    @staticmethod
    def calculate_maintainability_index(code: str, complexity: int) -> float:
        """Calculate maintainability index (0-100)"""
        lines = code.split('\n')
        loc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        if loc == 0:
            return 100.0
        
        # Simplified Halstead volume estimation
        operators = len(re.findall(r'[+\-*/=<>!&|^~%]', code))
        operands = len(re.findall(r'\b\w+\b', code))
        volume = (operators + operands) * (1 + (operators + operands) / 10) if operators + operands > 0 else 1
        
        # Maintainability Index formula (simplified)
        mi = 171 - 5.2 * (volume ** 0.23) - 0.23 * complexity - 16.2 * (loc ** 0.25)
        
        # Normalize to 0-100
        return max(0, min(100, mi))
    
    @staticmethod
    def check_security(code: str, language: Language = Language.PYTHON) -> List[CodeIssue]:
        """Check for security vulnerabilities"""
        issues = []
        lang_key = language.value if language.value in SECURITY_PATTERNS else "python"
        patterns = SECURITY_PATTERNS.get(lang_key, {})
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, (message, severity) in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        message=message,
                        severity=severity,
                        category=IssueCategory.SECURITY,
                        line_number=i,
                        code_snippet=line.strip()[:50]
                    ))
        
        return issues
    
    @staticmethod
    def check_style(code: str, max_line_length: int = 88) -> List[CodeIssue]:
        """Check code style issues"""
        issues = []
        lines = code.split('\n')
        
        import_section_ended = False
        
        for i, line in enumerate(lines, 1):
            # Line length
            if len(line) > max_line_length:
                issues.append(CodeIssue(
                    message=f"Line too long ({len(line)} > {max_line_length})",
                    severity=Severity.LOW,
                    category=IssueCategory.STYLE,
                    line_number=i,
                    suggestion=f"Break line into multiple lines"
                ))
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(CodeIssue(
                    message="Trailing whitespace",
                    severity=Severity.INFO,
                    category=IssueCategory.STYLE,
                    line_number=i
                ))
            
            # Import ordering
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('import') and not stripped.startswith('from'):
                import_section_ended = True
            
            if import_section_ended and (stripped.startswith('import ') or stripped.startswith('from ')):
                issues.append(CodeIssue(
                    message="Import not at top of file",
                    severity=Severity.LOW,
                    category=IssueCategory.STYLE,
                    line_number=i,
                    suggestion="Move imports to the top of the file"
                ))
            
            # Multiple statements on one line
            if ';' in line and not line.strip().startswith('#'):
                issues.append(CodeIssue(
                    message="Multiple statements on one line",
                    severity=Severity.LOW,
                    category=IssueCategory.STYLE,
                    line_number=i,
                    suggestion="Split into separate lines"
                ))
        
        return issues
    
    @staticmethod
    def check_documentation(code: str) -> List[CodeIssue]:
        """Check documentation issues"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for docstring
                    has_docstring = (node.body and 
                                    isinstance(node.body[0], ast.Expr) and
                                    isinstance(node.body[0].value, ast.Constant) and
                                    isinstance(node.body[0].value.value, str))
                    
                    if not has_docstring and not node.name.startswith('_'):
                        issues.append(CodeIssue(
                            message=f"Function '{node.name}' missing docstring",
                            severity=Severity.LOW,
                            category=IssueCategory.DOCUMENTATION,
                            line_number=node.lineno,
                            suggestion="Add a docstring describing the function's purpose"
                        ))
                
                elif isinstance(node, ast.ClassDef):
                    has_docstring = (node.body and 
                                    isinstance(node.body[0], ast.Expr) and
                                    isinstance(node.body[0].value, ast.Constant) and
                                    isinstance(node.body[0].value.value, str))
                    
                    if not has_docstring:
                        issues.append(CodeIssue(
                            message=f"Class '{node.name}' missing docstring",
                            severity=Severity.LOW,
                            category=IssueCategory.DOCUMENTATION,
                            line_number=node.lineno,
                            suggestion="Add a docstring describing the class"
                        ))
        except SyntaxError:
            pass
        
        return issues
    
    @staticmethod
    def check_best_practices(code: str) -> List[CodeIssue]:
        """Check for best practice violations"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Bare except
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append(CodeIssue(
                        message="Bare 'except:' clause",
                        severity=Severity.MEDIUM,
                        category=IssueCategory.BEST_PRACTICE,
                        line_number=node.lineno,
                        suggestion="Specify exception type, e.g., 'except Exception:'"
                    ))
                
                # Mutable default argument
                if isinstance(node, ast.FunctionDef):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            issues.append(CodeIssue(
                                message=f"Mutable default argument in '{node.name}'",
                                severity=Severity.MEDIUM,
                                category=IssueCategory.BEST_PRACTICE,
                                line_number=node.lineno,
                                suggestion="Use None as default and initialize inside function"
                            ))
                
                # Global statement
                if isinstance(node, ast.Global):
                    issues.append(CodeIssue(
                        message=f"Use of 'global' statement",
                        severity=Severity.LOW,
                        category=IssueCategory.BEST_PRACTICE,
                        line_number=node.lineno,
                        suggestion="Consider passing values as parameters instead"
                    ))
                
                # Star imports
                if isinstance(node, ast.ImportFrom) and any(alias.name == '*' for alias in node.names):
                    issues.append(CodeIssue(
                        message=f"Star import from '{node.module}'",
                        severity=Severity.MEDIUM,
                        category=IssueCategory.BEST_PRACTICE,
                        line_number=node.lineno,
                        suggestion="Import specific names instead of using *"
                    ))
        except SyntaxError:
            pass
        
        return issues

tools = CodeAnalysisTools()
print("âœ… Enhanced analysis tools initialized!")


# ============================================================
# CELL 9: Base Agent Class and Specialized Agents
# ============================================================

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, logger: ObservabilityLogger):
        self.name = name
        self.logger = logger
    
    @abstractmethod
    def analyze(self, code: str, language: Language) -> List[CodeIssue]:
        """Analyze code and return issues"""
        pass
    
    def log(self, action: str, details: Any = None):
        """Log an action"""
        self.logger.log_action(self.name, action, details)


class SyntaxAnalyzerAgent(BaseAgent):
    """Agent for syntax and AST analysis"""
    
    def __init__(self, logger: ObservabilityLogger):
        super().__init__("SyntaxAnalyzer", logger)
        self.tools = CodeAnalysisTools()
    
    def analyze(self, code: str, language: Language) -> List[CodeIssue]:
        self.log("Starting syntax analysis")
        issues = []
        
        if language == Language.PYTHON:
            ast_result = self.tools.analyze_ast(code)
            
            if not ast_result.get('valid'):
                issues.append(CodeIssue(
                    message=f"Syntax Error: {ast_result.get('error')}",
                    severity=Severity.CRITICAL,
                    category=IssueCategory.SYNTAX,
                    line_number=ast_result.get('line'),
                    column=ast_result.get('offset')
                ))
        
        self.log("Completed syntax analysis", {'issues_found': len(issues)})
        return issues


class SecurityAgent(BaseAgent):
    """Agent for security vulnerability detection"""
    
    def __init__(self, logger: ObservabilityLogger):
        super().__init__("SecurityAgent", logger)
        self.tools = CodeAnalysisTools()
    
    def analyze(self, code: str, language: Language) -> List[CodeIssue]:
        self.log("Starting security scan")
        issues = self.tools.check_security(code, language)
        self.log("Completed security scan", {'issues_found': len(issues)})
        return issues


class StyleAgent(BaseAgent):
    """Agent for code style checking"""
    
    def __init__(self, logger: ObservabilityLogger, max_line_length: int = 88):
        super().__init__("StyleAgent", logger)
        self.max_line_length = max_line_length
        self.tools = CodeAnalysisTools()
    
    def analyze(self, code: str, language: Language) -> List[CodeIssue]:
        self.log("Checking code style")
        issues = self.tools.check_style(code, self.max_line_length)
        self.log("Completed style check", {'issues_found': len(issues)})
        return issues


class ComplexityAgent(BaseAgent):
    """Agent for complexity analysis"""
    
    def __init__(self, logger: ObservabilityLogger, max_complexity: int = 10):
        super().__init__("ComplexityAgent", logger)
        self.max_complexity = max_complexity
        self.tools = CodeAnalysisTools()
    
    def analyze(self, code: str, language: Language) -> List[CodeIssue]:
        self.log("Analyzing complexity")
        issues = []
        
        if language == Language.PYTHON:
            complexity = self.tools.calculate_complexity(code)
            
            if complexity['cyclomatic'] > self.max_complexity:
                issues.append(CodeIssue(
                    message=f"High cyclomatic complexity: {complexity['cyclomatic']} (max: {self.max_complexity})",
                    severity=Severity.MEDIUM,
                    category=IssueCategory.COMPLEXITY,
                    suggestion="Consider breaking down into smaller functions"
                ))
            
            if complexity['max_nesting'] > 4:
                issues.append(CodeIssue(
                    message=f"Deep nesting detected: {complexity['max_nesting']} levels",
                    severity=Severity.MEDIUM,
                    category=IssueCategory.COMPLEXITY,
                    suggestion="Reduce nesting by using early returns or extracting functions"
                ))
        
        self.log("Completed complexity analysis", {'issues_found': len(issues)})
        return issues


class DocumentationAgent(BaseAgent):
    """Agent for documentation checking"""
    
    def __init__(self, logger: ObservabilityLogger):
        super().__init__("DocumentationAgent", logger)
        self.tools = CodeAnalysisTools()
    
    def analyze(self, code: str, language: Language) -> List[CodeIssue]:
        self.log("Checking documentation")
        issues = []
        
        if language == Language.PYTHON:
            issues = self.tools.check_documentation(code)
        
        self.log("Completed documentation check", {'issues_found': len(issues)})
        return issues


class BestPracticesAgent(BaseAgent):
    """Agent for best practices checking"""
    
    def __init__(self, logger: ObservabilityLogger):
        super().__init__("BestPracticesAgent", logger)
        self.tools = CodeAnalysisTools()
    
    def analyze(self, code: str, language: Language) -> List[CodeIssue]:
        self.log("Checking best practices")
        issues = []
        
        if language == Language.PYTHON:
            issues = self.tools.check_best_practices(code)
        
        self.log("Completed best practices check", {'issues_found': len(issues)})
        return issues

print("âœ… All specialized agents defined!")


# ============================================================
# CELL 10: Code Review Orchestrator - Enhanced
# ============================================================

class CodeReviewOrchestrator:
    """Orchestrates all agents for comprehensive code review"""
    
    def __init__(self, logger: ObservabilityLogger = None, memory: ReviewMemory = None):
        self.logger = logger or ObservabilityLogger()
        self.memory = memory or ReviewMemory()
        self.tools = CodeAnalysisTools()
        
        # Initialize all agents
        self.agents = [
            SyntaxAnalyzerAgent(self.logger),
            SecurityAgent(self.logger),
            StyleAgent(self.logger),
            ComplexityAgent(self.logger),
            DocumentationAgent(self.logger),
            BestPracticesAgent(self.logger)
        ]
        
        self.logger.log_action("Orchestrator", "Initialized", 
                               {'agents': [a.name for a in self.agents]})
    
    def review_code(self, code: str, language: Language = None) -> ReviewResult:
        """Perform comprehensive code review"""
        self.logger.start_timer("review")
        self.logger.log_action("Orchestrator", "Starting code review")
        
        # Detect language if not provided
        if language is None:
            language = LanguageDetector.detect(code)
            self.logger.log_action("Orchestrator", f"Detected language: {language.value}")
        
        # Compute code hash
        code_hash = CodeUtils.compute_hash(code)
        
        # Collect all issues from agents
        all_issues: List[CodeIssue] = []
        for agent in self.agents:
            try:
                issues = agent.analyze(code, language)
                all_issues.extend(issues)
            except Exception as e:
                self.logger.log_action(agent.name, f"Error: {str(e)}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(code, language)
        
        # Create result
        execution_time = self.logger.stop_timer("review")
        result = ReviewResult(
            code_hash=code_hash,
            language=language,
            timestamp=datetime.now().isoformat(),
            issues=all_issues,
            metrics=metrics,
            execution_time=execution_time
        )
        
        # Store in memory and update metrics
        self.memory.store_review(result)
        self.logger.update_metrics(result)
        
        self.logger.log_action("Orchestrator", "Review completed", 
                               {'total_issues': len(all_issues), 
                                'execution_time': f"{execution_time:.3f}s"})
        
        return result
    
    def _calculate_metrics(self, code: str, language: Language) -> CodeMetrics:
        """Calculate code metrics"""
        total, blank, comment = CodeUtils.count_lines(code)
        
        metrics = CodeMetrics(
            lines_of_code=total - blank - comment,
            blank_lines=blank,
            comment_lines=comment
        )
        
        if language == Language.PYTHON:
            complexity = self.tools.calculate_complexity(code)
            metrics.cyclomatic_complexity = complexity['cyclomatic']
            metrics.cognitive_complexity = complexity['cognitive']
            metrics.nesting_depth = complexity['max_nesting']
            metrics.maintainability_index = self.tools.calculate_maintainability_index(
                code, complexity['cyclomatic']
            )
            
            ast_result = self.tools.analyze_ast(code)
            if ast_result.get('valid'):
                metrics.function_count = len(ast_result.get('functions', []))
                metrics.class_count = len(ast_result.get('classes', []))
        
        return metrics
    
    def print_report(self, result: ReviewResult):
        """Print a formatted review report"""
        print("\n" + "="*70)
        print("ğŸ“‹ CODE REVIEW REPORT")
        print("="*70)
        
        summary = result.get_summary()
        
        # Header
        print(f"\nğŸ“… Timestamp: {result.timestamp}")
        print(f"ğŸ”¤ Language: {result.language.value.upper()}")
        print(f"ğŸ”‘ Code Hash: {result.code_hash}")
        print(f"â�±ï¸� Execution Time: {result.execution_time:.3f}s")
        
        # Metrics
        print("\n" + "-"*70)
        print("ğŸ“Š CODE METRICS")
        print("-"*70)
        m = result.metrics
        print(f"  Lines of Code: {m.lines_of_code}")
        print(f"  Cyclomatic Complexity: {m.cyclomatic_complexity}")
        print(f"  Cognitive Complexity: {m.cognitive_complexity}")
        print(f"  Maintainability Index: {m.maintainability_index:.1f}/100 (Grade: {m.get_grade()})")
        print(f"  Functions: {m.function_count} | Classes: {m.class_count}")
        print(f"  Max Nesting Depth: {m.nesting_depth}")
        
        # Summary
        print("\n" + "-"*70)
        print("ğŸ“ˆ SUMMARY")
        print("-"*70)
        print(f"  Total Issues: {summary['total_issues']}")
        print(f"  Risk Level: {summary['risk_level']}")
        print(f"  Quality Grade: {summary['quality_grade']}")
        
        # Issues by severity
        print("\n  Issues by Severity:")
        for sev in Severity:
            count = summary['by_severity'][sev.value]
            if count > 0:
                print(f"    {SEVERITY_COLORS[sev]} {sev.value.upper()}: {count}")
        
        # Detailed issues
        if result.issues:
            print("\n" + "-"*70)
            print("ğŸ”� DETAILED ISSUES")
            print("-"*70)
            
            # Group by category
            by_category = {}
            for issue in result.issues:
                cat = issue.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(issue)
            
            for category, issues in by_category.items():
                print(f"\n  [{category.upper()}]")
                for issue in issues[:5]:  # Limit to 5 per category
                    print(f"    {issue}")
                    if issue.suggestion:
                        print(f"      ğŸ’¡ {issue.suggestion}")
                if len(issues) > 5:
                    print(f"    ... and {len(issues) - 5} more")
        else:
            print("\nâœ… No issues found! Code looks clean.")
        
        # AI Suggestions
        if result.ai_suggestions:
            print("\n" + "-"*70)
            print("ğŸ¤– AI SUGGESTIONS")
            print("-"*70)
            print(result.ai_suggestions)
        
        print("\n" + "="*70)

# Initialize orchestrator
orchestrator = CodeReviewOrchestrator(logger, memory)
print("âœ… Orchestrator ready! All agents initialized.")


# ============================================================
# CELL 11: Gemini AI Integration for Intelligent Suggestions
# ============================================================

class GeminiCodeReviewer:
    """Gemini AI-powered code reviewer for intelligent suggestions"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model = None
        self.available = False
        
        if GENAI_AVAILABLE and API_CONFIGURED:
            try:
                self.model = genai.GenerativeModel(model_name)
                self.available = True
                print(f"âœ… Gemini model '{model_name}' initialized!")
            except Exception as e:
                print(f"âš ï¸� Failed to initialize Gemini: {e}")
    
    def get_suggestions(self, code: str, issues: List[CodeIssue], 
                        language: Language = Language.PYTHON) -> Optional[str]:
        """Get AI-powered suggestions for fixing issues"""
        if not self.available:
            return None
        
        # Format issues for prompt
        issues_text = "\n".join([
            f"{i+1}. [{issue.severity.value.upper()}] {issue.message}" + 
            (f" (Line {issue.line_number})" if issue.line_number else "")
            for i, issue in enumerate(issues[:10])  # Limit to 10 issues
        ])
        
        prompt = f"""You are an expert {language.value} code reviewer. Analyze this code and provide specific fixes.

## Code:
```{language.value}
{code}
```

## Issues Detected:
{issues_text}

## Your Task:
1. Explain each issue briefly (1-2 sentences)
2. Provide corrected code snippets
3. Give best practice recommendations

Keep your response concise and actionable. Focus on the most critical issues first.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting AI suggestions: {str(e)}"
    
    def explain_code(self, code: str, language: Language = Language.PYTHON) -> Optional[str]:
        """Get AI explanation of what the code does"""
        if not self.available:
            return None
        
        prompt = f"""Explain what this {language.value} code does in simple terms:

```{language.value}
{code}
```

Provide:
1. A brief summary (2-3 sentences)
2. Key functions/classes and their purposes
3. Any potential issues or improvements
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def suggest_refactoring(self, code: str, language: Language = Language.PYTHON) -> Optional[str]:
        """Get AI suggestions for refactoring"""
        if not self.available:
            return None
        
        prompt = f"""Suggest refactoring improvements for this {language.value} code:

```{language.value}
{code}
```

Focus on:
1. Code organization and structure
2. Naming conventions
3. Design patterns that could be applied
4. Performance optimizations

Provide specific, actionable suggestions with code examples.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize Gemini reviewer
gemini_reviewer = GeminiCodeReviewer()


# ============================================================
# CELL 12: ADK Agent Integration (if available)
# ============================================================

if ADK_AVAILABLE:
    def analyze_code_tool(code: str) -> dict:
        """Tool function for ADK agent to analyze code"""
        result = orchestrator.review_code(code)
        return {
            "status": "success" if not result.issues else "issues_found",
            "total_issues": len(result.issues),
            "issues": [str(i) for i in result.issues[:10]],
            "metrics": result.metrics.to_dict(),
            "risk_level": result.get_summary()['risk_level']
        }
    
    code_review_agent = LlmAgent(
        name="code_review_agent",
        model=Gemini(model="gemini-2.0-flash"),
        instruction="""You are an expert code reviewer. Your job is to:
        1. Analyze code for bugs, security issues, and style problems
        2. Use the analyze_code_tool to scan the code
        3. Provide clear, actionable feedback
        4. Suggest specific improvements with examples
        
        Always be constructive and educational in your feedback.
        Prioritize security issues, then bugs, then style.
        """,
        tools=[analyze_code_tool],
    )
    
    print("âœ… ADK Code Review Agent created!")
    print(f"ğŸ¤– Agent: {code_review_agent.name}")
else:
    print("âš ï¸� ADK not available. Using standalone review system.")


# ============================================================
# CELL 13: Interactive Review Function
# ============================================================

def review_code_interactive(code: str, 
                            include_ai: bool = True,
                            language: Language = None,
                            verbose: bool = True) -> ReviewResult:
    """
    Perform an interactive code review with optional AI suggestions.
    
    Args:
        code: The code to review
        include_ai: Whether to include Gemini AI suggestions
        language: Programming language (auto-detected if None)
        verbose: Whether to print the full report
    
    Returns:
        ReviewResult object with all findings
    """
    print("\n" + "ğŸ”„"*30)
    print("\nğŸš€ Starting Code Review...\n")
    
    # Perform review
    result = orchestrator.review_code(code, language)
    
    # Get AI suggestions if requested and issues found
    if include_ai and result.issues and gemini_reviewer.available:
        print("\nğŸ¤– Getting AI-powered suggestions...")
        result.ai_suggestions = gemini_reviewer.get_suggestions(
            code, result.issues, result.language
        )
    
    # Print report if verbose
    if verbose:
        orchestrator.print_report(result)
    
    return result

print("âœ… Interactive review function ready!")
print("\nğŸ“– Usage: result = review_code_interactive(your_code)")


# ============================================================
# CELL 14: Demo - Sample Code Reviews
# ============================================================

# Sample 1: Code with security issues
sample_security_issues = '''
import os
import pickle

def run_command(cmd):
    os.system(cmd)  # Security issue!
    return True

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)  # Unsafe deserialization

password = "super_secret_123"  # Hardcoded password!

def process_input():
    user_input = input("Enter command: ")
    eval(user_input)  # Critical security issue!
'''

# Sample 2: Code with style and complexity issues
sample_style_issues = '''
def calculate_something(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
    x = 1; y = 2; z = 3
    very_long_variable_name_that_exceeds_the_maximum_line_length_limit_and_should_be_shortened = 42
    return x + y + z

from os import *

def no_docstring_function(x):
    return x * 2
'''

# Sample 3: Clean code example
sample_clean_code = '''
"""Module for mathematical operations."""

from typing import List, Optional


def calculate_average(numbers: List[float]) -> Optional[float]:
    """Calculate the average of a list of numbers.
    
    Args:
        numbers: A list of numbers to average.
        
    Returns:
        The average value, or None if the list is empty.
    """
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.history: List[float] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers and store in history."""
        result = a + b
        self.history.append(result)
        return result
'''

print("âœ… Sample code snippets ready!")
print("\nğŸ“� Available samples:")
print("  - sample_security_issues: Code with security vulnerabilities")
print("  - sample_style_issues: Code with style and complexity problems")
print("  - sample_clean_code: Well-written clean code example")


# ============================================================
# CELL 15: Run Demo - Security Issues Review
# ============================================================

print("\n" + "="*70)
print("ğŸ”’ DEMO 1: SECURITY ISSUES REVIEW")
print("="*70)

result1 = review_code_interactive(sample_security_issues, include_ai=True)


# ============================================================
# CELL 16: Run Demo - Style Issues Review
# ============================================================

print("\n" + "="*70)
print("ğŸ�¨ DEMO 2: STYLE & COMPLEXITY ISSUES REVIEW")
print("="*70)

result2 = review_code_interactive(sample_style_issues, include_ai=True)


# ============================================================
# CELL 17: Run Demo - Clean Code Review
# ============================================================

print("\n" + "="*70)
print("âœ¨ DEMO 3: CLEAN CODE REVIEW")
print("="*70)

result3 = review_code_interactive(sample_clean_code, include_ai=False)


# ============================================================
# CELL 18: View Metrics Summary
# ============================================================

logger.print_metrics_summary()

print("\nğŸ“Š Memory Statistics:")
stats = memory.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\nğŸ”� Most Common Issues:")
for issue, count in memory.get_common_issues(5):
    print(f"  [{count}x] {issue}")


# ============================================================
# CELL 19: Custom Code Review - Try Your Own Code!
# ============================================================

# Paste your code here to review it
your_code = '''
# Paste your Python code here
def example():
    pass
'''

# Uncomment the line below to review your code
# result = review_code_interactive(your_code, include_ai=True)

print("ğŸ“� To review your own code:")
print("1. Replace the 'your_code' variable with your code")
print("2. Uncomment the review_code_interactive line")
print("3. Run this cell")


# ============================================================
# CELL 20: Export Results
# ============================================================

def export_results_to_json(results: List[ReviewResult], filename: str = "review_results.json"):
    """Export review results to JSON file"""
    data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_reviews": len(results),
        "reviews": [r.to_dict() for r in results],
        "aggregate_metrics": logger.get_metrics()
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"âœ… Results exported to {filename}")
    return filename

# Export all demo results
all_results = [result1, result2, result3]
export_results_to_json(all_results)


# ============================================================
# CELL 21: Final Summary
# ============================================================

print("\n" + "="*70)
print("ğŸ�‰ AI CODE REVIEW AGENT - READY!")
print("="*70)
print("""
âœ… System Components:
   â€¢ Multi-Agent Architecture (6 specialized agents)
   â€¢ Security Vulnerability Detection
   â€¢ Code Quality Metrics
   â€¢ Style & Best Practices Checking
   â€¢ Gemini AI Integration
   â€¢ Memory & Learning System
   â€¢ Comprehensive Reporting

ğŸ“Š Capabilities:
   â€¢ Python code analysis (primary)
   â€¢ JavaScript/TypeScript detection
   â€¢ Cyclomatic & Cognitive complexity
   â€¢ Maintainability Index calculation
   â€¢ AI-powered fix suggestions

ğŸš€ Usage:
   result = review_code_interactive(your_code)

ğŸ“– See documentation above for more details.
""")
print("="*70)

