# Whisper large-v3-turbo inference test

`whisper_inference.ipynb` transcribes speech with `openai/whisper-large-v3-turbo` from:

1. **Datasets:** 10 English LibriSpeech clips and 10 **Tagalog/Filipino** FLEURS clips with reference transcripts (word error rate),
   plus a 62 s clip with timestamps
2. **Your microphone:** records, detects the language and transcribes
3. **Your own files:** anything in `my_audio/` (wav, mp3, m4a, flac, ogg)

Inference only. LoRA fine-tuning needs a GPU and is not included.

## Results (CPU)

- English WER 3.9%, Tagalog WER 13.3% (language detected correctly as `tl` on all 10 clips)
- **large-v3-turbo does not translate:** it was fine-tuned on transcription only, so `task="translate"` returns the original
  language. Use `openai/whisper-large-v3` or `medium` for speech → English translation.

## CPU

Runs on CPU in float32 (~3.5 GB RAM). On this laptop:
- a clip of up to 30 s takes ~20–25 s, because Whisper always processes a full 30 s window
- long audio runs at ~1.4× real time
- float32 was 2× faster than bfloat16 here

Word-level timestamps ran out of memory on the 7 GB laptop; segment timestamps work. For speed, change `MODEL_ID` to
`openai/whisper-small` or run on a GPU (float16 is used automatically).

## Environment (already created on this machine as `asr`)

```bash
conda create -y -n asr python=3.12 pulseaudio ffmpeg       # parecord/pactl for the mic, ffmpeg for mp3/m4a
conda activate asr
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate soundfile jiwer scipy pandas pyarrow matplotlib huggingface_hub jinja2 ipykernel
python -m ipykernel install --user --name asr --display-name "Python (asr)"
```

## Microphone on WSL

WSLg forwards the Windows microphone to Linux as the PulseAudio source `RDPSource`. Check it with:

```bash
conda activate asr
pactl list short sources          # should list RDPSource
```

If recordings are silent, allow microphone access in Windows: *Settings → Privacy & security → Microphone*.
On native Linux, set `MIC_DEVICE = None` in the Settings cell to use the default input.

## Run

Open `whisper_inference.ipynb`, pick the **Python (asr)** kernel and Run All. The microphone cell starts recording
as soon as it runs, so start speaking right away. Re-run it, or call `mic_demo(15)`, for more recordings.

## Outputs

- `outputs/tagalog_transcripts.csv`: Tagalog transcript, reference, WER, detected language
- `outputs/librispeech_transcripts.csv`: transcript, reference, WER and time per clip
- `outputs/long_62s_segments.csv`: start/end timestamps and text per segment
- `outputs/mic_recording.wav`: your latest microphone recording
- `outputs/my_audio_transcripts.csv`: transcripts of your own files
- `outputs/summary.csv`: WER and real-time factor

## Data

- `data/librispeech_01..10.wav`: `hf-internal-testing/librispeech_asr_dummy` (LibriSpeech clean validation, CC BY 4.0)
- `data/long_62s.wav`: `distil-whisper/librispeech_long`
- `data/tagalog_01..10.wav`: Google FLEURS `fil_ph` test set (CC BY 4.0), first 10 sentences of 4–20 s, streamed from `data/fil_ph/audio/test.tar.gz`
