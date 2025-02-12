import pygame
import pygame_widgets
import pygame_widgets.slider
import pygame_widgets.textbox
from typing import List
from .settings import BEIGE, BLACK, RED, WHITE

def create_slider(
    screen: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    min_val: int,
    max_val: int,
    step: int,
    initial: int
) -> pygame_widgets.slider.Slider:
    return pygame_widgets.slider.Slider(
        screen, 
        x, 
        y, 
        width, 
        height, 
        min=min_val, 
        max=max_val, 
        step=step,
        initial=initial,
        colour=BLACK,
        handleColour=RED,
        valueColour=WHITE
    )

def create_textbox(
    screen: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    font_size: int,
    colour : int
) -> pygame_widgets.textbox.TextBox:
    return pygame_widgets.textbox.TextBox(
        screen, 
        x, 
        y, 
        width,
        height, 
        fontSize=font_size,
        colour=colour,
        borderThickness=0
    )

def create_probability_textboxes(
    screen: pygame.Surface,
    num_classes: int = 10
) -> List[pygame_widgets.textbox.TextBox]:
    prob_textboxes = []
    for i in range(num_classes):
        prob_textboxes.append(
            pygame_widgets.textbox.TextBox(
                screen, 
                925, 
                250 + (i * 50), 
                175, 
                30, 
                fontSize=20,
                borderThickness=0,
                textColour=BLACK,
                colour=BEIGE
            )
        )
        prob_textboxes[i].disable()  # Disable editing
    return prob_textboxes
