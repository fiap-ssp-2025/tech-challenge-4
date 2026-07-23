"""Saudação pura — sem I/O."""


class EmptyNameError(ValueError):
    """Nome ausente ou só com espaços."""


def greet(name: str) -> str:
    """Retorna a saudação no formato fixo da spec.

    Raises:
        EmptyNameError: se `name` for vazio ou só whitespace.
    """
    cleaned = name.strip()
    if not cleaned:
        raise EmptyNameError("name must not be empty")
    return f"Hello, {cleaned}!"
