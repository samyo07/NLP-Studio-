import gradio as gr

from src.ui import build_interface


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
