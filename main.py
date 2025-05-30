
from dashscope.audio.asr import *
from dashscope.audio.tts_v2 import *
from speechSynthesizer import SpeechSynthesizerHandler,speechSynthesizerCallback
from voiceRecognizer import VoiceRecognizerHandler,voiceRecognizerCallback
from kimi import ChatBotHandler,RobotWrapper
from main_sam_ur import VLMBot_sam
from realsense import realsense_camera

if __name__ == '__main__':
    chatBot = ChatBotHandler()
    foo = VLMBot_sam()
    camera = realsense_camera()
    speechSynthesizer= SpeechSynthesizerHandler()
    voiceRecognizer = VoiceRecognizerHandler()
    for i in range(3):
        print("-----------------------")
        camera.get_image()
        user_voice = voiceRecognizer.forward()
        if user_voice:
            image_Path = "outputs/color.png"
            print(user_voice)
            chatBot_response = chatBot.forward_image(image_Path, user_voice)

            print(chatBot_response)
            response = chatBot.extract_content(chatBot_response,"response")
            code = chatBot.extract_content(chatBot_response,"code")

            speechSynthesizer.forward(response)
            speechSynthesizer.readSpeech()

            print(response)
            print(code)
            exec(code)
            plan()



