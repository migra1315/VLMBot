import pyaudio
import dashscope
from dashscope.audio.asr import *
from dashscope.audio.tts_v2 import *
from openai import OpenAI
from datetime import datetime
import json

# 若没有将API Key配置到环境变量中，需将your-api-key替换为自己的API Key
dashscope.api_key = "sk-9c0fea83ac074bcbab481a077d74c83b"

class voiceRecognizerCallback(TranslationRecognizerCallback):
    def __init__(self, recognizer):
        self.recognizer = recognizer

    def on_open(self) -> None:
        # global mic
        # global stream
        print("TranslationRecognizerCallback open.")
        self.recognizer.mic = pyaudio.PyAudio()
        self.recognizer.stream = self.recognizer.mic.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True
        )
        print("open mic")

    def on_close(self) -> None:
        # global mic
        # global stream
        print("TranslationRecognizerCallback close.")
        if self.recognizer.stream:  # 添加检查
            self.recognizer.stream.stop_stream()
            self.recognizer.stream.close()
        if self.recognizer.mic:  # 添加检查
            self.recognizer.mic.terminate()
        self.recognizer.stream = None
        self.recognizer.mic = None

        # self.recognizer.stream.stop_stream()
        # self.recognizer.stream.close()
        # self.recognizer.mic.terminate()
        # self.recognizer.stream = None
        # self.recognizer.mic = None

    def on_event(
        self,
        request_id,
        transcription_result: TranscriptionResult,
        translation_result: TranslationResult,
        usage,
    ) -> None:
        # print("request id: ", request_id)
        # print("usage: ", usage)
        if translation_result is not None:
            print(
                "translation_languages: ",
                translation_result.get_language_list(),
            )
            english_translation = translation_result.get_translation("en")
            print("sentence id: ", english_translation.sentence_id)
            print("translate to english: ", english_translation.text)
            if english_translation.vad_pre_end:
                print("vad pre end {}, {}, {}".format(transcription_result.pre_end_start_time, transcription_result.pre_end_end_time, transcription_result.pre_end_timemillis))
        if transcription_result is not None:
            # print("sentence id: ", transcription_result.sentence_id)
            # print("transcription: ", transcription_result.text)
            self.recognizer.result = transcription_result.text


class VoiceRecognizerHandler():
    def __init__(self):  
        pass

    def forward(self):
        self.callback = voiceRecognizerCallback(self)
        self.translator = TranslationRecognizerChat(
            model="gummy-chat-v1",
            format="pcm",
            sample_rate=16000,
            transcription_enabled=True,
            translation_enabled=False,
            translation_target_languages=["en"],
            callback=self.callback,
        )
        self.result = ""
        self.mic = None
        self.stream = None  # 使用实例变量存储资源
    
        # 明确重置资源状态
        if self.stream:
            self.stream.close()
        if self.mic:
            self.mic.terminate()
        self.stream = None
        self.mic = None
        self.result=None
        
        self.translator.start()
        print("请您通过麦克风讲话体验一句话语音识别和翻译功能")
        while True:
            if self.stream:
                data = self.stream.read(3200, exception_on_overflow=False)
                if not self.translator.send_audio_frame(data):
                    print("sentence end, stop sending")
                    break
            else:
                print("something wrong")
                break

        self.translator.stop()
        return self.result

