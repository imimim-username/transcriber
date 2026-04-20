def diarize(audio_path):

    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    import torch
    import pandas as pd

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="")

    pipeline.to(torch.device("cuda"))

    with ProgressHook() as hook:
        # run the pipeline on an audio file
        diarization = pipeline(audio_path, hook=hook)

    diarized_path = "/home/imimim/gits/transcriber/audio/diarized.rttm"
    
    with open(diarized_path, "w") as rttm:
        diarization.write_rttm(rttm)
    
    #convert diarized audio to datafram and then dict

    df = pd.read_csv(diarized_path, sep=' ', header=None)

    diarization_data = []

    for index, row in df.iterrows():
        start_time = float(row[3])
        end_time = float(row[3]) + float(row[4])
        speaker = row[7]
    segment = {
        'start_time' : start_time,
        'end_time' : end_time,
        'speaker' : speaker
    }
    
    diarization_data.append(segment)
    
    return diarization_data