import pygame
import torch
from .settings import WIDTH, HEIGHT, RED, BLACK, BEIGE, GRAY
from .ui_elements import (
    create_slider, 
    create_textbox, 
    create_probability_textboxes
)
from model_pipeline.model_utils import load_model
from .utils import grayscale_controller


class AppSetup:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Live MNIST Fashion Classifier")

        self.CAPTURE_EVENT = pygame.USEREVENT + 1
        pygame.time.set_timer(self.CAPTURE_EVENT, 1000, loops=0)

        self.canvas_surface = pygame.Surface((WIDTH // 2, HEIGHT))
        self.prediction_surface = pygame.Surface((WIDTH // 2, HEIGHT))

        self.canvas_surface.fill(BLACK)
        self.prediction_surface.fill(BEIGE)

        self.slider = create_slider(
            self.screen, 
            x=1200, 
            y=392, 
            width=256, 
            height=20, 
            min_val=0, 
            max_val=255, 
            step=1,
            initial=255
        )
        self.output = create_textbox(
            self.screen, 
            x=1300, 
            y=422, 
            width=75, 
            height=40, 
            font_size=25,
            colour=BEIGE
        )
        self.prob_textboxes = create_probability_textboxes(self.screen)

        self.mouse_pressed = False
        self.model = load_model()

        self.probabilities_cpu = torch.zeros(10)
        self.current_widths = torch.zeros(10)
        self.target_widths = torch.zeros(10)

        self.running = True

        self.draw_header()
        self.set_grayscale, self.get_grayscale = grayscale_controller(255)

    def cleanup(self):
        pygame.quit()

    def get_events(self) -> list[pygame.event.Event]:
        return pygame.event.get()

    def get_screen(self) -> pygame.Surface:
        return self.screen

    def get_canvas_surface(self) -> pygame.Surface:
        return self.canvas_surface

    def get_prediction_surface(self) -> pygame.Surface:
        return self.prediction_surface

    def get_slider(self):
        return self.slider

    def get_output(self):
        return self.output

    def get_prob_textboxes(self):
        return self.prob_textboxes

    def draw_header(self):
        font = pygame.font.SysFont('Roboto Medium', 40)
        text_surface = font.render(
            "Welcome to Live MNIST Fashion Classifier!", 
            True, 
            GRAY
        )
        text_rect = text_surface.get_rect()

        right_panel_x = WIDTH // 2  
        panel_width = WIDTH // 2  

        text_rect.center = (right_panel_x + panel_width // 2, 50)

        self.screen.blit(text_surface, text_rect)
        
        instruction_font = pygame.font.SysFont('Roboto', 24)
        instructions = [
            "Instructions:",
            "- Draw a fashion item on the left side",
            "- The model will predict it every second",
            "- Press 'C' to clear the canvas",
            "- Adjust grayscale with the slider",
        ]

        y_offset = text_rect.bottom + 20
        for instruction in instructions:
            instruction_surface = instruction_font.render(
                instruction, 
                True, 
                RED
            )
            instruction_rect = instruction_surface.get_rect(
                center=(right_panel_x + panel_width // 2, y_offset)
            )
            self.screen.blit(instruction_surface, instruction_rect)
            y_offset += 30
