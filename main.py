from pydub import AudioSegment

from diarize import diarize

audiofile_path = "/home/imimim/gits/transcriber/audio/Imim-Gorby.m4a"

wav_path = "/home/imimim/gits/transcriber/audio/Imim-Gorby.wav"

#convert to .wav
audio = AudioSegment.from_file(audiofile_path)
# Export the audio as a WAV file
audio.export(wav_path, format="wav")

diarized_segments = diarize(wav_path)

print(diarized_segments)