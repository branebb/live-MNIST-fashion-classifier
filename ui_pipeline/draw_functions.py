import pygame
from typing import Callable
from .settings import BLACK, SCALE, GRID_SIZE
from PIL import Image

def draw_on_canvas(mouse_pos: tuple[int, int], canvas_surface: pygame.Surface, 
                   grayscale_value: int) -> None:
    grid_x = mouse_pos[0] // SCALE
    grid_y = mouse_pos[1] // SCALE

    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
        pygame.draw.rect(
            canvas_surface,
            (grayscale_value, grayscale_value, grayscale_value),
            (grid_x * SCALE, grid_y * SCALE, SCALE, SCALE)
        )

def clear_canvas(canvas_surface: pygame.Surface) -> None:
    canvas_surface.fill(BLACK)

def get_pixel_values(
        canvas_surface: pygame.Surface, 
        transform: Callable[[Image.Image], any]
    ) -> any:
    pixel_data = pygame.surfarray.array3d(canvas_surface)
    pixel_values = pixel_data[:, :, 0]
    downscaled_values = pixel_values[::SCALE, ::SCALE]
    downscaled_values_transpose = downscaled_values.T

    img = Image.fromarray(
        downscaled_values_transpose.astype('uint8'), 
        mode='L'
    )
    transformed_value = transform(img)
    return transformed_value
