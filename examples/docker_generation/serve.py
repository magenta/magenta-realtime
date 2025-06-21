import base64
from dataclasses import asdict
import io
from typing import Optional

import fastapi
import numpy as np
import uvicorn

from magenta_rt import system, audio

_APP = fastapi.FastAPI()
_SYSTEM: Optional[system.MagentaRTBase] = None

@_APP.get("/system_config")
async def system_config():
    assert _SYSTEM is not None
    return asdict(_SYSTEM.config)


@_APP.post("/embed_style")
async def embed_style(request: fastapi.Request):
    assert _SYSTEM is not None
    data = await request.json()
    if "audio" in data:
        audio_bytes = base64.b64decode(data["audio"])
        style_input = audio.Waveform.from_file(io.BytesIO(audio_bytes))
    elif "text" in data:
        style_input = data["text"]
    else:
        raise ValueError("Invalid request")
    return {"style_embedding": _SYSTEM.embed_style(style_input).tolist()}


@_APP.post("/generate_chunk")
async def generate_chunk(request: fastapi.Request):
    assert _SYSTEM is not None
    data = await request.json()
    
    # Convert style_embedding from list back to numpy array if present
    if "style_embedding" in data and data["style_embedding"] is not None:
        data["style_embedding"] = np.array(data["style_embedding"], dtype=np.float32)
    
    # Convert state from dict back to MagentaRTState if present
    state = None
    if "state" in data and data["state"] is not None:
        data["state"] = system.MagentaRTState.from_dict(_SYSTEM.config, data["state"])

    # Generate
    audio, state = _SYSTEM.generate_chunk(**data)

    # Prepare response
    audio_bytes = io.BytesIO()
    audio.write(audio_bytes, format="WAV")
    audio_b64 = base64.b64encode(audio_bytes.getvalue()).decode("utf-8")
    return {
        "audio_b64": audio_b64, 
        "state": state.to_dict(),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", default=False, help="Use mock system")
    args = parser.parse_args()

    if args.mock:
        _SYSTEM = system.MockMagentaRT()
    else:
        _SYSTEM = system.MagentaRT(lazy=False)
    uvicorn.run(_APP, host="0.0.0.0", port=8000)