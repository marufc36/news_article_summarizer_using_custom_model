from model import NewsSummaryModel
import gradio as gr
from transformers import T5TokenizerFast


tokenizer = T5TokenizerFast.from_pretrained("t5-base")
best_model = NewsSummaryModel.load_from_checkpoint("best-checkpoint.ckpt")
best_model.freeze()


def encode_text(text):
    encoding = tokenizer.encode_plus(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoding["input_ids"], encoding["attention_mask"]

def generate_summary(input_ids, attention_mask, model):
    model = model.to(input_ids.device)
    generated_ids = model.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    return generated_ids

def decode_summary(generated_ids):
    summary = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
               for gen_id in generated_ids]
    return "".join(summary)

def summarize(text):
    input_ids, attention_mask = encode_text(text)
    generated_ids = generate_summary(input_ids, attention_mask, best_model)
    summary = decode_summary(generated_ids)
    return summary

# Create Gradio interface
input_text = gr.Textbox(lines=10, label="Input Text")
output_text = gr.Textbox(label="Summary")

gr.Interface(
    fn=summarize,
    inputs=input_text,
    outputs=output_text,
    title="News Summary App",
    description="Enter a news text and get its summary."
).launch()
