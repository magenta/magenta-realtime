import base64
import requests
import io

from tqdm import tqdm

from magenta_rt import audio

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--text_style", type=str, default=None)
    parser.add_argument("--audio_style", type=str, default=None)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    server_url = args.server_url

    # Get system config
    config = requests.get(f"{server_url}/system_config").json()

    # Embed style
    if args.text_style is not None:
        output_filename = f"{args.text_style}.wav"
        style_embedding = requests.post(
            f"{server_url}/embed_style",
            json={"text": args.text_style}
        ).json()["style_embedding"]
    elif args.audio_style is not None:
        output_filename = "audio.wav"
        with open(args.audio_style, "rb") as f:
            style_embedding = requests.post(
                f"{server_url}/embed_style",
                json={"audio_b64": base64.b64encode(f.read()).decode("utf-8")}
            ).json()["style_embedding"]
    else:
        raise ValueError("Either --text_style or --audio_style must be provided")

    # Generate audio
    num_chunks = round(args.duration / config["chunk_length"])
    state = None
    audio_chunks = []
    for i in tqdm(range(num_chunks)):
        response = requests.post(f"{server_url}/generate_chunk", json={"style_embedding": style_embedding, "state": state}).json()
        audio_b64 = response["audio_b64"]
        state = response["state"]
        
        # Decode base64 audio and create Waveform object
        audio_bytes = base64.b64decode(audio_b64)
        audio_chunk = audio.Waveform.from_file(io.BytesIO(audio_bytes))
        audio_chunks.append(audio_chunk)

    audio_result = audio.concatenate(audio_chunks)
    audio_result.write(output_filename)
    print(f"Generated audio saved to {output_filename}")