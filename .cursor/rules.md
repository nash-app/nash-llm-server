# Project Rules

## Dependency Management

This project uses `pyproject.toml` for dependency management instead of `requirements.txt`. This is the modern Python packaging standard that provides better dependency resolution and project metadata management.

### Adding New Dependencies

When adding new dependencies to the project:

1. Add them to the `[project.dependencies]` section in `pyproject.toml`
2. Use the `>=` operator to specify minimum versions
3. Run `poetry install` to update the lockfile and install dependencies

Example:

```toml
[project.dependencies]
fastapi = ">=0.68.0"
uvicorn = ">=0.15.0"
python-dotenv = ">=0.19.0"
litellm = ">=1.0.0"
helicone = ">=0.1.0"
```

### Development Dependencies

Development dependencies should be added to the `[tool.poetry.group.dev.dependencies]` section:

```toml
[tool.poetry.group.dev.dependencies]
pytest = ">=7.0.0"
black = ">=22.0.0"
isort = ">=5.0.0"
```

### Installing Dependencies

To install all dependencies:

```bash
poetry install
```

To install with development dependencies:

```bash
poetry install --with dev
```
