# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""WebSocket-based serving backend for Magenta RealTime."""

import asyncio
import enum
import functools
import json
import logging
import mimetypes
import os
import struct
import time
from typing import Any

from absl import app
from absl import flags
from aiohttp import web
import librosa
import numpy as np

from . import path as path_lib
from . import system as system_lib

_PORT = flags.DEFINE_integer(
    'port',
    8000,
    'Port to listen on.',
)
_DEMO = flags.DEFINE_enum(
    'demo',
    'helloworld',
    [
        'helloworld',
    ],
    'Which demo to use.',
)
_TAG = flags.DEFINE_enum(
    'tag',
    'large',
    ['large', 'base', 'mock'],
    'Tag of the model to use (use "mock" for testing).',
)
_SIMULATE_LATENCY = flags.DEFINE_float(
    'simulate_latency',
    0.0,
    'Additional simulated latency in seconds.',
)
_DEVICE = flags.DEFINE_string(
    'device',
    'gpu',
    'Device to use.',
)


def _get_static_asset(path: str, mode: str = 'r') -> str | bytes:
  """Returns the contents of a static asset file."""
  demo_dir = path_lib.MODULE_DIR.parent / 'demos' / _DEMO.value
  with open(demo_dir / path, mode) as f:
    return f.read()




class PlaybackState(enum.Enum):
  """State management for Magenta RT."""

  STOPPED = 'STOPPED'
  PAUSED = 'PAUSED'
  PLAYING = 'PLAYING'

  @classmethod
  def values(cls):
    return [item.value for item in cls]


_EMA_ALPHA = 0.5
_ADAPTIVE_INTERVAL_FACTOR = 1.05


def _parse_kwargs(kwargs):
  """Parse keyword arguments for generation."""
  if not isinstance(kwargs, dict):
    return {}
  parsed = kwargs.copy()
  if 'guidance_weight' in parsed:
    parsed['guidance_weight'] = float(parsed['guidance_weight'])
  if 'temperature' in parsed:
    parsed['temperature'] = float(parsed['temperature'])
  if 'topk' in parsed:
    parsed['topk'] = int(round(parsed['topk']))
  return parsed


class Server:
  """WebSocket server for Magenta RealTime."""

  def __init__(
      self,
      system: system_lib.MagentaRTBase,
      port: int = 8000,
      simulate_latency: float = 0.0,
  ):
    # Core state.
    self._system = system
    self._port = port
    self._chunk_length = self._system.chunk_length
    self._session_ws: web.WebSocketResponse | None = None
    self._simulate_latency = simulate_latency

    # Playback state.
    self._playback_state: PlaybackState | None = None
    self._adaptive_playback = False
    self._chunk_generation_time_ema: float | None = None

    # Generation state.
    self._system_state = self._system.init_state()
    self._style_embedding: np.ndarray | None = None
    self._generation_kwargs = {}

    # AsyncIO state
    self._stream_lock = asyncio.Lock()
    self._generation_task: asyncio.Task | None = None

    # Web server.
    self._app = web.Application()
    self._app.router.add_get('/stream', self.handle_ws)
    self._app.router.add_get('/stream_info', self.handle_get_stream_info)
    self._app.router.add_post('/style', self.handle_post_style)
    self._app.router.add_get('/{path:.*}', self.handle_get_static_file)

  @property
  def session_is_active(self) -> bool:
    """Returns whether a session is currently active."""
    return self._session_ws is not None

  def _update_ema(self, sample: float):
    """Update exponential moving average of chunk generation time."""
    if self._chunk_generation_time_ema is None:
      self._chunk_generation_time_ema = sample
    else:
      self._chunk_generation_time_ema = (
          _EMA_ALPHA * sample
          + (1 - _EMA_ALPHA) * self._chunk_generation_time_ema
      )

  def _update_control_values(self, body: dict[str, Any]):
    """Updates style and kwargs from message body, assuming lock is held."""
    if not body:
      return
    style = body.get('style')
    gen_kwargs = body.get('generation_kwargs')
    if style is not None:
      style_dim = self._system.config.style_embedding_dim
      try:
        emb = np.array(style, dtype=np.float32)
        if emb.shape == (style_dim,):
          self._style_embedding = emb
        else:
          logging.error(
              'Invalid style embedding shape: %s, expected (%d,)',
              emb.shape,
              style_dim,
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('Failed to parse style embedding: %s', e)
    if gen_kwargs is not None:
      self._generation_kwargs.update(_parse_kwargs(gen_kwargs))

  async def _run_in_executor(self, fn, *args, **kwargs):
    """Run blocking function in executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, functools.partial(fn, *args, **kwargs)
    )

  async def _generate_chunk_in_bg(self):
    """Generate chunk in background thread, return samples."""
    async with self._stream_lock:
      style = self._style_embedding
      kwargs = self._generation_kwargs
      state = self._system_state

    t_start = time.perf_counter()
    try:
      chunk, new_state = await self._run_in_executor(
          self._system.generate_chunk, state=state, style=style, **kwargs
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Error in chunk generation executor: %s', e)
      await self._handle_ws_end_session()
      return None

    if self._simulate_latency > 0.0:
      logging.debug('Simulating latency of %.3fs', self._simulate_latency)
      await asyncio.sleep(self._simulate_latency)

    async with self._stream_lock:
      self._system_state = new_state

    t_gen = time.perf_counter() - t_start
    logging.debug('Chunk generation time: %.3fs', t_gen)
    self._update_ema(t_gen)

    return chunk.samples

  async def _send_chunk(self, chunk_samples: np.ndarray):
    """Send chunk samples over websocket."""
    try:
      audio_floats = [float(s) for s in chunk_samples.reshape(-1)]
      buf = struct.pack(f'<{len(audio_floats)}f', *audio_floats)
      async with self._stream_lock:
        if self._session_ws and not self._session_ws.closed:
          await self._session_ws.send_bytes(buf)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Failed to send chunk: %s', e)
      await self._handle_ws_end_session()

  async def _generation_loop(self):
    """Background task for generating and sending audio chunks."""
    clean_stop = False
    loop = asyncio.get_running_loop()
    try:
      # The first chunk is sent by _handle_start_session.
      # This loop generates chunk i, then waits until it's time to send it.
      next_send_time = loop.time() + self._chunk_length
      while True:
        async with self._stream_lock:
          current_state = self._playback_state

        if current_state is None or current_state == PlaybackState.STOPPED:
          logging.info(
              'Playback state is %s, ending generation loop.', current_state
          )
          clean_stop = True
          break

        while current_state == PlaybackState.PAUSED:
          await asyncio.sleep(0.01)
          async with self._stream_lock:
            current_state = self._playback_state
          if current_state is None or current_state == PlaybackState.STOPPED:
            break
        if current_state is None or current_state == PlaybackState.STOPPED:
          logging.info(
              'Playback state is %s, ending generation loop.', current_state
          )
          clean_stop = True
          break

        logging.info('Generating chunk')
        chunk_samples = await self._generate_chunk_in_bg()
        if chunk_samples is None:  # Error during generation
          break

        async with self._stream_lock:
          adaptive = self._adaptive_playback
          current_ema = self._chunk_generation_time_ema

        if adaptive and current_ema:
          target_interval = current_ema * _ADAPTIVE_INTERVAL_FACTOR
        else:
          target_interval = self._chunk_length

        if adaptive and abs(target_interval - self._chunk_length) > 1e-6:
          rate = self._chunk_length / target_interval
          logging.debug(
              'Time stretching chunk by %.3f to fit interval %.3fs',
              rate,
              target_interval,
          )
          chunk_samples = await self._run_in_executor(
              librosa.effects.time_stretch,
              y=chunk_samples.astype(np.float32),
              rate=rate,
          )

        # Wait until it's time to send this chunk.
        now = loop.time()
        wait_time = next_send_time - now
        if wait_time > 0:
          await asyncio.sleep(wait_time)
        elif wait_time < -0.1:  # If we are lagging significantly
          logging.warning('Falling behind realtime by %.3fs', -wait_time)
          # Skip missed intervals to catch up to current time + interval.
          next_send_time = now + target_interval

        # Update interval for next chunk's send time and schedule it.
        async with self._stream_lock:
          current_state = self._playback_state
        next_send_time += target_interval

        if current_state == PlaybackState.PLAYING:
          await self._send_chunk(chunk_samples)

    except asyncio.CancelledError:
      logging.info('Generation loop cancelled.')
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('Error in generation loop: %s', e)
    finally:
      if not clean_stop and self.session_is_active:
        await self._handle_ws_end_session()

  async def _handle_ws_start_session(
      self, ws: web.WebSocketResponse, body: dict[str, Any]
  ):
    """Handle StartSession message."""
    del body
    async with self._stream_lock:
      if self.session_is_active:
        raise RuntimeError(
            'A session is already active. Only one session is allowed at a'
            ' time.'
        )
      logging.info('Starting new session.')
      self._session_ws = ws
      self._playback_state = PlaybackState.STOPPED
      self._adaptive_playback = False
      self._chunk_generation_time_ema = None
      self._system_state = self._system.init_state()
      self._style_embedding = None
      self._generation_kwargs = {}

  async def _handle_ws_update_playback(self, body: dict[str, Any]):
    """Handle UpdatePlayback message."""
    if not body:
      return

    should_start_generation_loop = False
    async with self._stream_lock:
      state = body.get('state')
      if state in PlaybackState.values():
        new_state = PlaybackState(state)
        if new_state == PlaybackState.STOPPED:
          self._system_state = self._system.init_state()
        if new_state == PlaybackState.PLAYING and (
            self._generation_task is None or self._generation_task.done()
        ):
          if self._style_embedding is None:
            raise RuntimeError(
                'Style embedding must be set via UpdateControl before playing.'
            )
          should_start_generation_loop = True
        self._playback_state = new_state
        logging.info('Playback state set to: %s', self._playback_state)
      adaptive = body.get('adaptive')
      if adaptive is not None:
        self._adaptive_playback = bool(adaptive)
        logging.info('Adaptive playback set to: %s', self._adaptive_playback)

    if should_start_generation_loop:
      logging.info('First play: generating initial chunk and starting loop.')
      initial_chunk_samples = await self._generate_chunk_in_bg()
      if initial_chunk_samples is None:
        return
      await self._send_chunk(initial_chunk_samples)
      async with self._stream_lock:
        if self._playback_state == PlaybackState.PLAYING and (
            self._generation_task is None or self._generation_task.done()
        ):
          self._generation_task = asyncio.create_task(self._generation_loop())

  async def _handle_ws_update_control(self, body: dict[str, Any]):
    """Handle UpdateControl message."""
    if not body:
      return
    async with self._stream_lock:
      self._update_control_values(body)

  async def _handle_ws_received_chunk(self, body: dict[str, Any]):
    """Handle ReceivedChunk message."""
    del body
    # Could be used for more advanced adaptive timing based on RTT.
    logging.debug('ReceivedChunk ack')

  async def _handle_ws_end_session(self, body: dict[str, Any] | None = None):
    """Handle EndSession message or clean up after error/disconnect."""
    del body
    task_to_await = None
    async with self._stream_lock:
      if not self.session_is_active:
        return
      logging.info('Ending session.')
      self._playback_state = None
      if self._generation_task:
        self._generation_task.cancel()
        task_to_await = self._generation_task
        self._generation_task = None
      if self._session_ws and not self._session_ws.closed:
        await self._session_ws.close()
      self._session_ws = None

    if task_to_await:
      try:
        await task_to_await
      except asyncio.CancelledError:
        pass

  async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
    """Handles WebSocket connections."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_id = f'{request.remote}'
    logging.info('WebSocket connection established: %s', client_id)

    async with self._stream_lock:
      if self.session_is_active:
        logging.warning(
            'New connection attempt from %s while session is active.',
            client_id,
        )
        await ws.close(message=b'Session already active')
        return ws

    try:
      async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
          try:
            data = json.loads(msg.data)
            msg_type = data.get('type')
            body = data.get('body')
            logging.debug('Received message: type=%s', msg_type)

            if msg_type == 'StartSession':
              await self._handle_ws_start_session(ws, body)
            elif msg_type == 'UpdatePlayback':
              await self._handle_ws_update_playback(body)
            elif msg_type == 'UpdateControl':
              await self._handle_ws_update_control(body)
            elif msg_type == 'ReceivedChunk':
              await self._handle_ws_received_chunk(body)
            elif msg_type == 'EndSession':
              await self._handle_ws_end_session(body)
            else:
              logging.warning('Unknown message type: %s', msg_type)

          except json.JSONDecodeError:
            logging.error('Failed to decode JSON message: %s', msg.data)
          except Exception as e:  # pylint: disable=broad-exception-caught
            logging.exception('Error processing message: %s', e)
            await self._handle_ws_end_session()
            break

        elif msg.type == web.WSMsgType.ERROR:
          logging.error(
              'WS connection for %s closed with exception %s',
              client_id,
              ws.exception(),
          )
          await self._handle_ws_end_session()
          break
    finally:
      logging.info('WebSocket connection closed: %s', client_id)
      # If this client was the active session, end it.
      is_session_ws = False
      async with self._stream_lock:
        if self._session_ws is ws:
          is_session_ws = True
      if is_session_ws:
        await self._handle_ws_end_session()
    return ws

  async def handle_get_static_file(self, request: web.Request) -> web.Response:
    """Handles HTTP GET requests for static files."""
    path = request.path
    if not path.startswith('/'):
      return web.Response(status=400, text=f'Invalid argument: {path}')
    if path == '/':
      path = '/index.html'

    relative_path = path.lstrip('/')
    for component in relative_path.split('/'):
      if not component or component == '.' or component == '..':
        return web.Response(status=400, text=f'Invalid argument: {path}')

    try:
      data = _get_static_asset(relative_path, mode='rb')
    except FileNotFoundError:
      return web.Response(status=404, text=f'Not found: {path}')
    mimetype = mimetypes.guess_type(relative_path)[0]
    return web.Response(
        body=data,
        status=200,
        content_type=mimetype or 'application/octet-stream',
    )

  async def handle_get_stream_info(self, request: web.Request) -> web.Response:
    """Handles HTTP GET requests for stream info."""
    del request
    return web.json_response(
        {
            'chunk_length': self._chunk_length,
            'sample_rate': self._system.sample_rate,
            'num_channels': self._system.num_channels,
        },
        status=200,
        headers={'Access-Control-Allow-Origin': '*'},
    )

  async def handle_post_style(self, request: web.Request) -> web.Response:
    """Tokenizes text into style embeddings."""
    # TODO(chrisdonahue): Add support for audio file uploads.
    text = await request.text()
    logging.info('Received text: %s', text)
    embeddings = self._system.style_model.embed(text)
    return web.json_response(
        [float(f) for f in embeddings.reshape(-1)],
        status=200,
        headers={'Access-Control-Allow-Origin': '*'},
    )

  def run(self):
    """Runs the server."""
    logging.info('Starting server on port %d', self._port)
    web.run_app(self._app, port=self._port)


def main(_):
  if _TAG.value == 'mock':
    system = system_lib.MockMagentaRT(synthesis_type='sine')
  else:
    system = system_lib.MagentaRT(
        tag=_TAG.value,
        device=_DEVICE.value,
        lazy=False,
    )
  server = Server(
      system=system,
      port=_PORT.value,
      simulate_latency=_SIMULATE_LATENCY.value,
  )
  server.run()


if __name__ == '__main__':
  app.run(main)
