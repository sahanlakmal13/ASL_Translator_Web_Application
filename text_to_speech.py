from gtts import gTTS
import os

language = "en"

def run_voice(text):
    output = gTTS(text=text, lang=language, slow=False)
    output.save("static/output.mp3")

    return os.system("start static/output.mp3")
