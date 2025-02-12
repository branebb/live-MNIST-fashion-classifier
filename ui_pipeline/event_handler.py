import pygame
import torch
from .draw_functions import (
    draw_on_canvas, 
    clear_canvas, 
    get_pixel_values
)
from .transformations import transformation
from concurrent.futures import ThreadPoolExecutor
from typing import List


class EventHandler:
    def __init__(self, app):
        self.app = app
        self.mouse_pressed = False
        self.executor = ThreadPoolExecutor(max_workers=4)

    def handle_events(self, events: List[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self.app.running = False
            elif event.type == self.app.CAPTURE_EVENT:
                self.handle_capture_event()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_pressed = True
                    self.handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_pressed:
                    self.handle_mouse_motion(event)
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)

        self.update_grayscale()

    def handle_capture_event(self) -> None:
        def run_prediction() -> None:
            try:
                to_predict = get_pixel_values(
                    self.app.get_canvas_surface(), 
                    transformation
                )
                _, probabilities = self.app.model.predict(to_predict)
                self.app.probabilities_cpu = probabilities.cpu()
                self.app.target_widths = (
                    self.app.probabilities_cpu.clone() * 100
                )
            except Exception as e:
                print(f"Prediction error: {e}")

        self.executor.submit(run_prediction)

    def handle_mouse_down(self, event: pygame.event.Event) -> None:
        draw_on_canvas(
            event.pos, 
            self.app.get_canvas_surface(), 
            self.app.get_grayscale()
        )

    def handle_mouse_motion(self, event: pygame.event.Event) -> None:
        draw_on_canvas(
            event.pos, 
            self.app.get_canvas_surface(), 
            self.app.get_grayscale()
        )

    def handle_keydown(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_c:
            clear_canvas(self.app.get_canvas_surface())
            self.app.target_widths = torch.zeros(10)

    def update_grayscale(self) -> None:
        self.app.set_grayscale(self.app.get_slider().getValue())
        self.app.get_output().setText(str(self.app.get_grayscale()))
