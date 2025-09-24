
# Gemini CLI Customization (`gemini.md`)

This file helps customize the Gemini CLI for the specific needs of this project.

## Project Overview

- **Name:** Nubank Foundation Models
- **Description:** This project explores the use of foundation models at Nubank.
- **Primary Language:** Python
- **Key Technologies:** TensorFlow, PyTorch, scikit-learn

## File and Directory Exclusions

The following files and directories should be ignored by Gemini to avoid processing unnecessary or sensitive information.

- `**/node_modules/`
- `**/dist/`
- `*.log`
- `secrets.json`

## Preferred Libraries and Conventions

When generating code, please adhere to the following conventions:

- **Python Style:** Follow the PEP 8 style guide. Use `black` for formatting.
- **API Framework:** Use FastAPI for building new services.
- **Testing:** Use `pytest` for all new tests.

## Custom Commands and Aliases

Here are some project-specific commands that are frequently used:

- **`run_tests`**: `pytest -v`
- **`start_server`**: `uvicorn main:app --reload`
- **`lint`**: `ruff check .`

## Project-Specific Instructions

- When asked to create a new model, use the template from `templates/model_template.py`.
- Always include type hints in Python code.
- For any new API endpoint, ensure it is documented with an OpenAPI schema.

