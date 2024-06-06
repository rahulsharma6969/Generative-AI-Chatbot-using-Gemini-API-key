import google.generativeai as genai
import speech_recognition as sr
import os
import time
import warnings 
warnings.filterwarnings("ignore", message = r"torch.utils._pytree._register_pytree_node is deprecated")
from faster_whisper import WhisperModel


GOOGLE_API_KEY = '###'
genai.configure(api_key=GOOGLE_API_KEY)
whisper_size = 'base'
wake_word = 'gemini'
listening_for_wake_word = True
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device = 'cpu',
    compute_type = 'int8',
    cpu_threads = num_cores,
    num_workers = num_cores
    )

generation_config={
    "temperature":0.7,
    "top_p":1,
    "top_k":1,
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
model = genai.GenerativeModel('gemini-1.0-pro-latest', generation_config=generation_config,safety_settings=safety_settings)
convo=model.start_chat()

system_message = ''' INSTUCTIONS: Do not respond with anything but "AFFIRMATIVE." to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so. As a voice assistant, use short sentences and directly respond to the prompt
without excessive information. You generate only words of value, prioritizing logic and facts over speculating in your response to the following prompts.'''
system_message = system_message.replace(f'\n','')
convo.send_message(system_message)

r = sr.Recognizer()
source = sr.Microphone()

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def listen_for_wake_word(audio):
    global listening_for_wake_word
    wake_audio_path = 'wake_detect.wav'
    with open(wake_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    text_input = wav_to_text(wake_audio_path)
    if wake_word in text_input.lower().strip():
        print('Wake Word Detected. Please Speak your prompt to Gemini')
        listening_for_wake_word = False

def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)
        if len(prompt_text.strip())==0:
            print('Empty Prompt. Please speak again.')
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)
            convo.send_message(prompt_text)
            output = convo.last.text
            print('Gemini: ', output)
            print('\nSay', wake_word, 'to wake me up. \n')
            listening_for_wake_word = False

    except Exception as e:
        print('Prompt Error: ', e)
        

def callback(recognizer, audio):
    global listening_for_wake_word
    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print('\nSay', wake_word, 'to wake me up. \n')
    r.listen_in_background(source, callback)
    while True:
        time.sleep(0.5)



if __name__ == '__main__':
    start_listening()
