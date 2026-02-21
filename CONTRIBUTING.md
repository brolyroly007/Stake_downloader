# Contributing to Stake Downloader

Thanks for your interest in contributing! Here's how you can help.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Stake_downloader.git
   cd Stake_downloader
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
4. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Use **Black** for formatting: `black .`
- Use **Ruff** for linting: `ruff check .`
- Use **mypy** for type checking: `mypy .`
- Follow PEP 8 conventions
- Add type hints to function signatures
- Write docstrings for public functions and classes

### Project Conventions

- Use `async/await` for I/O-bound operations
- Use `loguru` logger instead of `print()` for production code
- Configuration goes through Pydantic settings (`core/config.py`)
- New modules should follow the existing pattern (base class + implementations)

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add proxy rotation support
fix: handle WebSocket reconnection on timeout
docs: update API endpoint documentation
refactor: extract download logic to separate module
test: add unit tests for virality scorer
```

### Testing

Run tests before submitting:

```bash
pytest tests/ -v
```

Add tests for new features in the `tests/` directory.

## Pull Request Process

1. Ensure your code passes all linting and tests
2. Update documentation if you changed any public API
3. Update the README if you added new features
4. Submit your PR with a clear description of the changes

## Areas for Contribution

- Improving test coverage
- Adding new stream platform monitors
- Enhancing video processing options
- Database persistence implementation
- Docker containerization
- Documentation improvements
- Bug fixes and error handling

## Questions?

Open an issue if you have questions or need help getting started.
