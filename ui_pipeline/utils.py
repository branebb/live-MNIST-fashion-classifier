from typing import Callable

def grayscale_controller(initial_value: int = 255) -> Callable[[], int]:
    grayscale_value = initial_value

    def set_grayscale(value: int) -> None:
        nonlocal grayscale_value
        grayscale_value = value

    def get_grayscale() -> int:
        return grayscale_value

    return set_grayscale, get_grayscale