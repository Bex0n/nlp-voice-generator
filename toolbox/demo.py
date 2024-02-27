import gradio as gr
import os

from datasets import load_dataset

"""
Load the VCTK dataset.
Create a light version of the dataset with only the audio and text columns.
"""
dataset = load_dataset("vctk", trust_remote_code=True)
dataset = dataset['train'].select(range(1952))

def fetch_item(speaker_id, text_id):
    for item in dataset:
        if item['speaker_id'] == speaker_id and item['text_id'] == text_id:
            return item
    return None

def process_demo(text_id, speaker_id, target_speaker_id):
    selected_text = fetch_item(speaker_id, text_id)['text']
    audio_path = fetch_item(speaker_id, text_id)['audio']['path']
    target_audio_path = fetch_item(target_speaker_id, text_id)['audio']['path']

    return selected_text, audio_path

# Unique text IDs for dropdown
text_ids = list(set(item['text_id'] for item in dataset))
# Unique speaker IDs for dropdown
speaker_ids = list(set(item['speaker_id'] for item in dataset))

text_id_select = gr.components.Dropdown(choices=text_ids, label="Select Text ID")
speaker_id_select = gr.components.Dropdown(choices=speaker_ids, label="Inference Speaker")
target_speaker_id_select = gr.components.Dropdown(choices=speaker_ids, label="Target Speaker")

demo = gr.Interface(fn=process_demo,
                    inputs=[text_id_select, speaker_id_select, target_speaker_id_select],
                    outputs=[gr.components.Textbox(label="Selected Text"), gr.components.Audio(label="Speaker Audio")],
                    title="Speaker Inference Demo",
                    description="Select an utterance and speakers for inference and target, and listen to the audio.")

if __name__ == "__main__":
    demo.launch()
