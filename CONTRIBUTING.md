# Contributing to LLaMA 3 Finance Robustness

Thank you for your interest in contributing! This project aims to provide a rigorous framework for evaluating LLM robustness in financial applications.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check existing [Issues](https://github.com/yourusername/llama3-finance-robustness/issues)
2. Create a new issue with:
   - Clear title
   - Detailed description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)

### Code Contributions

#### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/llama3-finance-robustness.git
   cd llama3-finance-robustness
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Coding Standards

**Style Guide**:
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use `black` for code formatting:
  ```bash
  black src/
  ```

**Docstrings**:
Use Google-style docstrings:
```python
def compute_robustness(entropy: float) -> float:
    """
    Compute robustness score from semantic entropy.

    Args:
        entropy: Semantic entropy value

    Returns:
        Robustness score in [0, 1]

    Raises:
        ValueError: If entropy is negative
    """
    pass
```

**Testing**:
- Write tests for new features
- Ensure all tests pass:
  ```bash
  pytest tests/
  ```
- Aim for >80% code coverage

#### Commit Guidelines

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Formatting changes
- `refactor:` Code restructuring
- `test:` Test additions/modifications
- `chore:` Maintenance tasks

Examples:
```bash
git commit -m "feat: add GPT-4 comparison module"
git commit -m "fix: correct entropy calculation for single cluster"
git commit -m "docs: update installation instructions for Windows"
```

#### Pull Request Process

1. **Update documentation**
   - Add/update docstrings
   - Update README if needed
   - Add to CHANGELOG.md

2. **Run checks**
   ```bash
   # Format code
   black src/

   # Run linter
   flake8 src/

   # Run tests
   pytest tests/

   # Type check (optional)
   mypy src/
   ```

3. **Submit PR**
   - Clear title describing the change
   - Reference related issues (#123)
   - Describe what changed and why
   - Include test results
   - Add screenshots for UI changes

4. **Code Review**
   - Address reviewer comments
   - Keep commits clean and logical
   - Be respectful and constructive

### Types of Contributions

#### 1. New Features

**Examples**:
- Support for additional LLMs (GPT-4, Claude, Gemini)
- New paraphrasing methods
- Alternative clustering algorithms
- Additional financial datasets
- Multi-language support

**Process**:
1. Open an issue to discuss the feature
2. Get approval from maintainers
3. Implement with tests
4. Submit PR

#### 2. Bug Fixes

**Examples**:
- Incorrect entropy calculations
- Memory leaks
- Incorrect documentation
- Installation issues

**Process**:
1. Verify the bug
2. Write a failing test (if applicable)
3. Fix the bug
4. Ensure test passes
5. Submit PR

#### 3. Documentation

**Examples**:
- Improve README
- Add tutorials
- Fix typos
- Add code examples
- Create video walkthroughs

**Process**:
1. Identify documentation gap
2. Write clear, concise content
3. Include code examples
4. Submit PR

#### 4. Performance Improvements

**Examples**:
- Optimize embedding computation
- Reduce memory usage
- Parallelize processing
- Cache intermediate results

**Process**:
1. Profile current performance
2. Implement optimization
3. Benchmark improvements
4. Document changes
5. Submit PR

### Development Workflow

#### Branch Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

#### Release Process

1. Create release branch: `release/v1.x.x`
2. Update version numbers
3. Update CHANGELOG.md
4. Create GitHub release
5. Merge to `main` and `develop`

### Testing Guidelines

**Unit Tests**:
```python
# tests/test_entropy.py
def test_entropy_single_cluster():
    """Test entropy calculation for single cluster."""
    labels = np.array([0, 0, 0, 0])
    entropy = compute_entropy(labels)
    assert entropy == 0.0

def test_entropy_equal_clusters():
    """Test entropy for equally distributed clusters."""
    labels = np.array([0, 0, 1, 1])
    entropy = compute_entropy(labels)
    assert np.isclose(entropy, 1.0)
```

**Integration Tests**:
```python
def test_full_pipeline():
    """Test complete pipeline execution."""
    # Load data
    # Generate prompts
    # Run model
    # Compute metrics
    # Assert expected outputs
    pass
```

### Code of Conduct

#### Our Standards

**Positive behaviors**:
- Using welcoming and inclusive language
- Respecting differing viewpoints
- Accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behaviors**:
- Trolling, insulting, or derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

#### Enforcement

Violations may result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to: [your.email@example.com]

### Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Credited in academic publications (if applicable)

### Questions?

- **General questions**: Open a [Discussion](https://github.com/yourusername/llama3-finance-robustness/discussions)
- **Bug reports**: Open an [Issue](https://github.com/yourusername/llama3-finance-robustness/issues)
- **Security concerns**: Email [your.email@example.com]

## Thank You!

Your contributions make this project better for everyone in the financial AI community.

---

**Happy Coding! 🚀**
