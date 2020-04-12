# ノイズ入りのデータを作成する

import subprocess

for light_idx in range(1,8):
    for volume in range(1,12):
        for noise_idx in range(1,11):
            l = str(light_idx)
            v_cmd = str(float(volume) / 10)
            v_name = str(volume)
            n = str(noise_idx)
            cmd = f'sox -m -v1.0 no_processing_sound/voice/light{l}_A4_UH.wav -v{v_cmd}  no_processing_sound/noise/noise{n}.wav learning_sample/light/{l}_{v_name}_{n}.wav trim 0 3'
            subprocess.check_output(cmd, shell=True)

for pull_idx in range(1,8):
    for volume in range(1,12):
        for noise_idx in range(1,11):
            l = str(pull_idx)
            v_cmd = str(float(volume) / 10)
            v_name = str(volume)
            n = str(noise_idx)
            cmd = f'sox -m -v1.0 no_processing_sound/voice/pull{l}_A4_UH.wav -v{v_cmd}  no_processing_sound/noise/noise{n}.wav learning_sample/pull/{l}_{v_name}_{n}.wav trim 0 3'
            subprocess.check_output(cmd, shell=True)