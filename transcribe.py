import os 
import speech_recognition as sr 
from pydub import AudioSegment

def prepare_audio_file(audio_path):
    audio_file= AudioSegment.from_file(audio_path,format='m4a')
    audio_file.export(audio_path[:-3]+"wav",format='wav')
    return audio_path[:-3]+"wav"

def transcribe_audio(audio_data):
    r = sr.Recognizer()
    text=r.recognize_google(audio_data,language="en-US")
    return text

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    audio_files = [f for f in os.listdir('.') if f.endswith('.m4a')]
    for audio_file_name in audio_files:
        with open(audio_file_name[:-3]+"txt","w") as f:
            f.write("")
        audio_path = prepare_audio_file(audio_file_name)
        audio_chunk=[]
        audio_file=AudioSegment.from_wav(audio_path)
        audio_file_length=len(audio_file)
        chunk_length=5000
        for i in range(0,audio_file_length,chunk_length):
            audio_chunk.append(audio_file[i:i+chunk_length])
        hour_start=0
        minute_start=0
        second_start=0
        line=""
        for i,chunk in enumerate(audio_chunk):
            chunk.export("chunk{0}.wav".format(i),format='wav')
            audio_path="chunk{0}.wav".format(i)
            with sr.AudioFile(audio_path) as source:
                audio_data = sr.Recognizer().record(source)
                try:
                    text=transcribe_audio(audio_data)
                except:
                    if line!="":
                        hour_end=(i*5)//3600
                        minute_end=(i*5)//60-hour_end*60
                        second_end=(i*5)%60
                        with open(audio_file_name[:-3]+"txt","a") as f:
                            f.write("{0:02d}:{1:02d}:{2:02d} {3:02d}:{4:02d}:{5:02d} ".format(hour_start,minute_start,second_start,hour_end,minute_end,second_end))
                            f.write(line+"\n")
                        line=""
                    continue
                else:
                    if line=="":
                        hour_start=(i*5)//3600
                        minute_start=(i*5)//60-hour_start*60
                        second_start=(i*5)%60
                    line+=text+" "
                finally:
                    os.remove("chunk{0}.wav".format(i))