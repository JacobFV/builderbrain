# CONTRIBUTING.md — builderbrain

## 0. purpose

Contributing guidelines for BuilderBrain development. This document defines code standards, review processes, testing requirements, and contribution workflows for both human developers and AI agents.

---

## 1. development philosophy

### 1.1 core principles

**composition over memorization:**
* Build reusable skills, not memorized responses
* Favor explicit planning over implicit patterns
* Design for generalization, not overfitting

**constraints over freedom:**
* Use formal grammars to shape behavior
* Apply safety invariants to prevent harm
* Balance creativity with reliability

**transparency over opacity:**
* Make plans auditable and verifiable
* Log decisions and constraint applications
* Enable debugging and analysis

**evolution over stagnation:**
* Continuous improvement through constraints
* Adaptive learning from feedback
* Systematic experimentation and validation

### 1.2 development mindset

**think in graphs:**
* Model problems as DAGs (nodes = skills, edges = dependencies)
* Design composable components
* Validate execution paths before implementation

**embrace constraints:**
* Constraints are features, not limitations
* Use grammars to compress search spaces
* Safety invariants prevent catastrophic failures

**measure everything:**
* Instrument all components for observability
* Track constraint satisfaction and violations
* Monitor performance and quality metrics

---

## 2. code standards

### 2.1 python style guide

**formatting:**
* Use black for code formatting (`black .`)
* Maximum line length: 88 characters
* Use type hints for all function signatures
* Docstrings for all public methods

**naming conventions:**
* `snake_case` for functions and variables
* `PascalCase` for classes
* `UPPER_CASE` for constants
* Descriptive names that explain purpose

**imports:**
* Standard library imports first
* Third-party imports second
* Local imports last
* No wildcard imports (`from module import *`)

### 2.2 documentation standards

**docstrings:**
```python
def example_function(param1: str, param2: int) -> Dict[str, Any]:
    """
    Brief description of function purpose.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When input validation fails
        RuntimeError: When execution fails

    Examples:
        >>> example_function("test", 42)
        {'result': 'success'}
    """
    pass
```

**module documentation:**
* Every module must have a purpose statement
* List key classes and functions
* Document any non-obvious behavior

### 2.3 type hints and validation

**comprehensive typing:**
* Use `typing` module for complex types
* Generic types for collections (`List[str]`, `Dict[str, Any]`)
* Union types for optional parameters (`Optional[str]`)
* Protocol definitions for interfaces

**runtime validation:**
* Input validation for public APIs
* Type checking where performance allows
* Clear error messages for invalid inputs

---

## 3. testing requirements

### 3.1 test categories

**unit tests:**
* Individual function and method testing
* Mock external dependencies
* Test edge cases and error conditions
* Coverage target: >90% for core modules

**integration tests:**
* Component interaction testing
* Real dependencies where possible
* End-to-end functionality validation
* Test data flow and constraint application

**system tests:**
* Full pipeline testing with real data
* Performance benchmarking
* Constraint satisfaction verification
* Error handling and recovery testing

**regression tests:**
* Prevent breaking changes
* Golden test cases for core functionality
* Performance regression detection
* Constraint violation monitoring

### 3.2 test structure

**test file organization:**
```
tests/
├── unit/
│   ├── test_math_utils.py
│   ├── test_dual_rail.py
│   └── test_grammar_parser.py
├── integration/
│   ├── test_training_pipeline.py
│   └── test_constraint_optimization.py
├── system/
│   ├── test_end_to_end.py
│   └── test_performance.py
└── golden/
    ├── grammar_tests.txt
    ├── plan_tests.yaml
    └── constraint_tests.json
```

**test naming:**
* `test_function_name` for function tests
* `test_class_name_method` for class method tests
* `test_integration_component_interaction` for integration tests

### 3.3 test data management

**synthetic test data:**
* Deterministic random seeds for reproducibility
* Domain-specific test scenarios
* Edge cases and adversarial inputs

**golden tests:**
* Reference outputs for critical functionality
* Grammar parsing validation
* Plan execution verification
* Constraint satisfaction baselines

---

## 4. contribution workflow

### 4.1 issue management

**issue types:**
* **Bug:** unexpected behavior or crashes
* **Feature:** new functionality requests
* **Enhancement:** improvements to existing features
* **Documentation:** updates to docs or comments
* **Refactoring:** code structure improvements

**issue format:**
```markdown
## Description
Brief description of the issue or feature

## Motivation
Why this change is needed

## Proposed Solution
How to implement the change

## Acceptance Criteria
What must be true for this to be complete

## Testing
How this will be tested and verified
```

### 4.2 pull request process

**pr requirements:**
* Clear title and description
* Reference to related issues
* Test coverage for new code
* Documentation updates if needed
* Performance impact assessment

**pr template:**
```markdown
## Summary
What this PR does and why

## Changes
- File1: description of changes
- File2: description of changes

## Testing
- Unit tests added/updated
- Integration tests verified
- Performance benchmarks run

## Documentation
- README updated if needed
- Docstrings added/updated
- Comments for complex logic

## Checklist
- [ ] Tests pass
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Performance verified
```

### 4.3 code review guidelines

**review criteria:**
* **Correctness:** does it work as intended?
* **Clarity:** is the code easy to understand?
* **Consistency:** follows established patterns?
* **Performance:** acceptable resource usage?
* **Safety:** no security vulnerabilities?
* **Maintainability:** easy to modify and extend?

**review process:**
1. Automated checks (linting, tests, formatting)
2. Peer review by domain experts
3. Integration testing in staging
4. Final approval and merge

---

## 5. ai agent contributions

### 5.1 agent capabilities

**permitted actions:**
* Code implementation and refactoring
* Test writing and maintenance
* Documentation updates
* Issue analysis and proposal
* Performance optimization

**restricted actions:**
* Direct production deployments
* Security-sensitive changes
* User data access or modification
* External service integrations
* Financial or legal decisions

### 5.2 agent contribution process

**proposal phase:**
1. Analyze issue requirements
2. Propose implementation approach
3. Estimate effort and complexity
4. Identify dependencies and risks

**implementation phase:**
1. Create feature branch with descriptive name
2. Implement solution with comprehensive tests
3. Update documentation and examples
4. Run full test suite and benchmarks

**review phase:**
1. Submit PR with detailed description
2. Respond to human reviewer feedback
3. Address any issues or concerns
4. Obtain approval before merge

### 5.3 agent-human collaboration

**communication expectations:**
* Clear explanations of implementation choices
* Documentation of complex logic or algorithms
* Identification of uncertainties or assumptions
* Requests for clarification when needed

**feedback loops:**
* Accept and incorporate human feedback
* Explain reasoning for design decisions
* Document lessons learned from iterations
* Maintain transparency about limitations

---

## 6. development environment

### 6.1 local setup

**prerequisites:**
* Python 3.11+
* Virtual environment (venv or conda)
* Git and GitHub CLI
* IDE with Python support (VSCode, PyCharm, etc.)

**installation:**
```bash
# Clone repository
git clone https://github.com/your-org/builderbrain.git
cd builderbrain

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync  # Install all dependencies including dev tools
# Or activate virtual environment (optional)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**development tools:**
* **Linting:** black, isort, flake8
* **Testing:** pytest, pytest-cov
* **Type checking:** mypy, pyright
* **Documentation:** sphinx, mkdocs

### 6.2 testing workflow

**local testing:**
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/

# Run with coverage
uv run pytest --cov=bb_core --cov-report=html

# Run performance tests
uv run pytest tests/system/test_performance.py --benchmark-only
```

**continuous integration:**
* Automated testing on every PR
* Performance regression detection
* Security vulnerability scanning
* Documentation building and validation

---

## 7. architecture guidelines

### 7.1 module organization

**separation of concerns:**
* **Core:** mathematical foundations, protocols, interfaces
* **Neural:** model architectures, training components
* **Runtime:** execution, validation, constraints
* **Infrastructure:** logging, monitoring, deployment

**dependency rules:**
* Runtime modules cannot import training modules
* Core modules cannot import neural modules
* All modules must be importable independently

### 7.2 interface design

**protocol definitions:**
* Use `Protocol` for interface definitions
* Runtime-checkable protocols where possible
* Clear contracts for component interactions

**error handling:**
* Specific exception types for different error categories
* Graceful degradation with fallbacks
* Comprehensive error messages for debugging

### 7.3 performance considerations

**efficiency requirements:**
* Inference latency: <100ms for simple queries
* Memory usage: <2GB for base models
* Throughput: >100 requests/second per GPU

**optimization strategies:**
* Batch processing for efficiency
* Caching for expensive operations
* Lazy loading for large components
* Profiling and performance monitoring

---

## 8. security guidelines

### 8.1 secure coding practices

**input validation:**
* Sanitize all user inputs
* Validate data types and ranges
* Prevent injection attacks
* Handle malformed data gracefully

**authentication and authorization:**
* Multi-factor authentication for admin access
* Role-based access control (RBAC)
* Audit logging for all privileged operations
* Secure credential storage

**data protection:**
* Encryption at rest and in transit
* PII detection and anonymization
* Secure deletion procedures
* Compliance with privacy regulations

### 8.2 security testing

**security test types:**
* Static analysis for vulnerabilities
* Dynamic testing for runtime issues
* Penetration testing for attack vectors
* Dependency vulnerability scanning

**security review:**
* All code changes reviewed for security implications
* Third-party security assessments
* Incident response planning and testing
* Security training for all contributors

---

## 9. deployment and operations

### 9.1 deployment pipeline

**stages:**
1. **Development:** local testing and validation
2. **Staging:** integration testing with real data
3. **Pre-production:** performance and load testing
4. **Production:** monitored rollout with rollback capability

**deployment checks:**
* All tests pass in target environment
* Performance benchmarks met
* Constraint compliance verified
* Documentation updated

### 9.2 operational procedures

**monitoring and alerting:**
* Real-time performance monitoring
* Automated alerting for anomalies
* Incident response procedures
* Post-mortem analysis for failures

**maintenance:**
* Regular dependency updates
* Performance optimization
* Security patch application
* Documentation maintenance

---

## 10. community guidelines

### 10.1 communication standards

**respectful collaboration:**
* Assume good intent in all interactions
* Focus on technical merits of contributions
* Provide constructive feedback
* Acknowledge others' contributions

**inclusive environment:**
* Welcome diverse perspectives and backgrounds
* Use inclusive language in code and documentation
* Support accessibility requirements
* Foster learning and growth

### 10.2 contribution recognition

**acknowledgment:**
* Credit contributors in release notes
* Maintain contributor lists in documentation
* Public recognition for significant contributions
* Opportunities for community leadership

**feedback:**
* Regular contributor surveys
* Open feedback channels
* Transparent decision-making processes
* Continuous improvement of contribution experience

---

## 11. future contributions

**priority areas:**
* Safety constraint improvements
* Multi-domain skill composition
* Advanced grammar formalisms
* Performance optimizations
* Documentation enhancements

**contribution opportunities:**
* Bug fixes and stability improvements
* New domain implementations
* Research prototype integrations
* Community tool development
* Educational content creation
