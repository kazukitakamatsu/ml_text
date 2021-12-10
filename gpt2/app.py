import gradio as gr
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

def generate_text(input_text, max_length):
    input = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input, do_sample=True, max_length=max_length, num_return_sequences=3)
    output_text = tokenizer.batch_decode(output)
    return output_text

inputs = [
  gr.inputs.Textbox(lines=1, placeholder="昔々あるところに"), 
  gr.inputs.Slider(minimum=10, maximum=100, default=30, label="文字数")
]
iface = gr.Interface(
  fn=generate_text, 
  inputs=inputs,
  outputs="textbox",
  server_name="0.0.0.0",
  server_port=8080
  )
iface.launch()
