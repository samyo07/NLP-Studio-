import gradio as gr

from .config import INTRO_TEXT
from .pipelines import (
    sentiment_single,
    sentiment_batch,
    emotion_single,
    toxicity_single,
    topic_classify,
    ner_single,
    summarize_text,
    detect_language,
)


def build_interface() -> gr.Blocks:
    css = """
    .result-card {
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        font-size: 14px;
    }
    .footer-note {
        margin-top: 16px;
        font-size: 12px;
        color: #6b7280;
    }
    """

    with gr.Blocks(title="NLP Studio", css=css) as demo:
        gr.Markdown(INTRO_TEXT)

        with gr.Tab("Sentiment"):
            with gr.Row():
                with gr.Column():
                    s_txt = gr.Textbox(
                        label="Text",
                        placeholder="Type or paste a sentence...",
                        lines=4,
                    )
                    s_explain = gr.Checkbox(
                        label="Highlight simple positive and negative words",
                        value=True,
                    )
                    s_btn = gr.Button("Analyze sentiment")
                with gr.Column():
                    s_html = gr.HTML(
                        "<div class='result-card'>Result will appear here.</div>"
                    )
                    s_chart = gr.HTML()
            s_btn.click(
                fn=sentiment_single,
                inputs=[s_txt, s_explain],
                outputs=[s_html, s_chart],
            )

            gr.Markdown("### Batch mode")
            b_txt = gr.Textbox(
                label="One text per line",
                placeholder="Line one\nLine two\nLine three",
                lines=6,
            )
            b_btn = gr.Button("Run batch sentiment")
            b_table = gr.Dataframe(
                headers=[
                    "text",
                    "predicted_label",
                    "confidence",
                    "negative",
                    "neutral",
                    "positive",
                ],
                datatype=["str", "str", "number", "number", "number", "number"],
                interactive=False,
                label="Batch results",
            )
            b_file = gr.File(label="Download CSV")
            b_btn.click(
                fn=sentiment_batch,
                inputs=[b_txt],
                outputs=[b_table, b_file],
            )

        with gr.Tab("Emotion"):
            e_txt = gr.Textbox(
                label="Text",
                placeholder="Write something with emotional tone...",
                lines=4,
            )
            e_explain = gr.Checkbox(
                label="Highlight simple positive and negative words",
                value=False,
            )
            e_btn = gr.Button("Analyze emotion")
            e_html = gr.HTML(
                "<div class='result-card'>Result will appear here.</div>"
            )
            e_chart = gr.HTML()
            e_btn.click(
                fn=emotion_single,
                inputs=[e_txt, e_explain],
                outputs=[e_html, e_chart],
            )

        with gr.Tab("Toxicity"):
            t_txt = gr.Textbox(
                label="Text",
                placeholder="Enter a sentence to check toxicity...",
                lines=4,
            )
            t_btn = gr.Button("Check toxicity")
            t_html = gr.HTML(
                "<div class='result-card'>Result will appear here.</div>"
            )
            t_chart = gr.HTML()
            t_btn.click(
                fn=toxicity_single,
                inputs=[t_txt],
                outputs=[t_html, t_chart],
            )

        with gr.Tab("Topics"):
            z_txt = gr.Textbox(
                label="Text",
                placeholder="Short paragraph to classify...",
                lines=4,
            )
            z_labels = gr.Textbox(
                label="Candidate labels (comma separated)",
                value="technology, sports, finance, politics, health, entertainment",
            )
            z_multi = gr.Checkbox(
                label="Allow multiple correct topics",
                value=True,
            )
            z_btn = gr.Button("Classify topics")
            z_html = gr.HTML(
                "<div class='result-card'>Result will appear here.</div>"
            )
            z_chart = gr.HTML()
            z_btn.click(
                fn=topic_classify,
                inputs=[z_txt, z_labels, z_multi],
                outputs=[z_html, z_chart],
            )

        with gr.Tab("Entities"):
            n_txt = gr.Textbox(
                label="Text",
                placeholder="Tim Cook met Sundar Pichai in Paris to discuss AI.",
                lines=4,
            )
            n_btn = gr.Button("Detect entities")
            n_html = gr.HTML(
                "<div class='result-card'>Result will appear here.</div>"
            )
            n_btn.click(
                fn=ner_single,
                inputs=[n_txt],
                outputs=[n_html],
            )

        with gr.Tab("Summary"):
            sum_txt = gr.Textbox(
                label="Long text",
                placeholder="Paste a long paragraph or article...",
                lines=8,
            )
            sum_words = gr.Slider(
                label="Target summary length (approx words)",
                minimum=30,
                maximum=200,
                value=80,
                step=10,
            )
            sum_btn = gr.Button("Summarize")
            sum_html = gr.HTML(
                "<div class='result-card'>Result will appear here.</div>"
            )
            sum_btn.click(
                fn=summarize_text,
                inputs=[sum_txt, sum_words],
                outputs=[sum_html],
            )

        with gr.Tab("Language"):
            l_txt = gr.Textbox(
                label="Text",
                placeholder="Write something in any language...",
                lines=4,
            )
            l_btn = gr.Button("Detect language")
            l_html = gr.HTML(
                "<div class='result-card'>Result will appear here.</div>"
            )
            l_chart = gr.HTML()
            l_btn.click(
                fn=detect_language,
                inputs=[l_txt],
                outputs=[l_html, l_chart],
            )

        gr.Markdown(
            "<div class='footer-note'>"
            "Models used: "
            "<code>cardiffnlp/twitter-roberta-base-sentiment-latest</code>, "
            "<code>j-hartmann/emotion-english-distilroberta-base</code>, "
            "<code>unitary/toxic-bert</code>, "
            "<code>dslim/bert-base-NER</code>, "
            "<code>facebook/bart-large-mnli</code>, "
            "<code>facebook/bart-large-cnn</code>, "
            "<code>papluca/xlm-roberta-base-language-detection</code>"
            "</div>"
        )

    return demo
