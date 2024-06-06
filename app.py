import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import os
import asyncio
import warnings
from io import BytesIO
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated")

GOOGLE_API_KEY = '###'
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 500
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

model = genai.GenerativeModel('gemini-1.0-pro-latest', generation_config=generation_config, safety_settings=safety_settings)
convo = model.start_chat()

# Whisper Model Configuration
whisper_size = 'base'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores,
    model_size_or_path=whisper_size
)

wake_word = 'gpt'
listening_for_wake_word = True

system_message = ''' INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE." to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so. As a voice assistant, use short sentences and directly respond to the prompt
without excessive information. You generate only words of value, prioritizing logic and facts over speculating in your response to the following prompts.'''
system_message = system_message.replace('\n', '')
convo.send_message(system_message)

r = sr.Recognizer()
source = sr.Microphone()

def wav_to_text(audio_data):
    try:
        audio_file = BytesIO(audio_data)
        segments, _ = whisper_model.transcribe(audio_file)
        text = ''.join(segment.text for segment in segments)
        return text
    except Exception as e:
        st.write(f"Error transcribing audio: {e}")
        return ""

def listen_for_wake_word(audio):
    global listening_for_wake_word
    try:
        audio_data = audio.get_wav_data()
        text_input = wav_to_text(audio_data)
        if wake_word in text_input.lower().strip():
            st.write('Wake Word detected. Please speak your prompt')
            listening_for_wake_word = False
    except Exception as e:
        st.write(f"Error processing wake word: {e}")

def prompt_gpt(audio):
    global listening_for_wake_word
    try:
        audio_data = audio.get_wav_data()
        prompt_text = wav_to_text(audio_data)
        if len(prompt_text.strip()) == 0:
            st.write('Empty Prompt, Speak Again')
            listening_for_wake_word = True
        else:
            st.write('User: ' + prompt_text)
            convo.send_message(prompt_text)
            output = convo.last.text
            st.write('Gemini: ' + output)
            st.write('\nSay', wake_word, 'to wake me up. \n')
            listening_for_wake_word = True

    except Exception as e:
        st.write(f"Prompt Error: {e}")

async def listen_loop():
    global listening_for_wake_word
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
        st.write('\nSay', wake_word, 'to wake me up. \n')
        while True:
            try:
                audio = r.listen(s)
                if listening_for_wake_word:
                    listen_for_wake_word(audio)
                else:
                    prompt_gpt(audio)
            except Exception as e:
                st.write(f"Listening error: {e}")
            await asyncio.sleep(0.5)

st.title("Voice Assistant")
if st.button('Start Listening'):
    asyncio.run(listen_loop())
