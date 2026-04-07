# Contributing to Enterprise Task Automation Environment

## Development Setup

```bash
# Clone and setup
git clone <repo>
cd enterprise_env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev]
```

## Code Style

We follow PEP 8 with Black formatting:

```bash
black src/ tests/
ruff check src/ tests/
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## Adding New Features

### New Action Types

1. Add to `ActionType` enum in `src/types.py`
2. Implement handler in `EnterpriseEnv._process_action()`
3. Update `_get_valid_actions()` if needed
4. Add tests in `tests/test_environment.py`
5. Document in `API_REFERENCE.md`

### New Task Difficulties

1. Create class inheriting from `TaskGrader` in `src/graders.py`
2. Implement `grade()` method
3. Register in `get_task_graders()`
4. Add tests and documentation

### New Metrics

1. Add to `Observation` model in `src/types.py`
2. Calculate in `_get_observation()`
3. Update graders if needed

## Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Format code: `black .` and `ruff check --fix .`
5. Run tests: `pytest tests/`
6. Push and create PR with description

## Issue Templates

Use these for reporting:

### Bug Report
```
**Describe the bug**
Brief description

**To Reproduce**
Steps to reproduce

**Expected behavior**
What should happen

**Actual behavior**
What happens instead
```

### Feature Request
```
**Describe the feature**
What should be added

**Justification**
Why is this useful

**Implementation approach**
Suggested approach (optional)
```

## Code Organization

```
src/
  __init__.py          # Package exports
  types.py             # All data models
  environment.py       # Main RL environment
  graders.py           # Task evaluation

tests/
  test_environment.py  # Environment tests
  test_graders.py      # Grader tests
  test_types.py        # Model tests
```

## Performance Guidelines

- Keep environment step time < 10ms
- Support environments with 1000+ emails/meetings without crash
- Memory usage should be < 500MB for standard configs
- Support up to 1000 parallel environments

## Documentation

- Document all public methods with docstrings
- Include type hints
- Add examples for complex features
- Update README.md for major changes

## Release Process

1. Update `__version__` in `src/__init__.py`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.2.0`
4. Push: `git push origin --tags`
5. Build: `python -m build`
6. Upload: `twine upload dist/*`

## Questions?

- Open an issue for questions
- Check existing issues for common problems
- Review documentation before asking

Thank you for contributing!
