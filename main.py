import pygame
import pygame_widgets
import sys
from ui_pipeline.app_setup import AppSetup
from ui_pipeline.event_handler import EventHandler
from ui_pipeline.settings import WHITE, GREEN, FASHION_LABELS


def run_app():
    app = AppSetup()
    event_handler = EventHandler(app)

    while app.running:
        app.get_screen().fill(WHITE)
        app.get_screen().blit(app.get_canvas_surface(), (0, 0))
        app.get_screen().blit(
            app.get_prediction_surface(), 
            (app.get_screen().get_width() // 2, 0)
        )

        app.draw_header()

        events = app.get_events()

        event_handler.handle_events(events)

        app.current_widths += (
            app.target_widths - app.current_widths
        ) * 0.05

        for i in range(10):
            pygame.draw.rect(
                app.get_screen(),
                GREEN,
                (800, 250 + i * 50, app.current_widths[i].item(), 30)
            )
            app.get_prob_textboxes()[i].setText(
                f"{FASHION_LABELS[i]}: " 
                f"{app.probabilities_cpu[i].item() * 100:.2f}%"
            )

        pygame_widgets.update(events)
        pygame.display.update()

    app.cleanup()
    sys.exit()


if __name__ == "__main__":
    run_app()