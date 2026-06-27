from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Self, Sequence, Union

from flax import struct
from jax import numpy as jnp, tree_util
import numpy as np
import soundfile

from .resample import resample


# todo: configure when initializing an AudioTree instance?
_str_max_length = 256


@struct.dataclass
class AudioTree:
    """
    A `flax.struct.dataclass`_ for holding audio information including a waveform, sample rate, and metadata.

    The ``AudioTree`` class is inspired by Descript AudioTools's `AudioSignal`_.
        .. _AudioSignal: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        .. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass

    Args:
        waveform (jnp.ndarray): Audio waveform data shaped ``(Samples)``, ``(Channels, Samples)``, or ``(Batch, Channels, Samples)``
        sample_rate (int): Sample rate of ``waveform``, such as 44100 Hz.
        loudness (jnp.ndarray, optional): Loudness of the audio waveform in LUFs. You may not need to set this when initializing. Instead,
            use ``replace_loudness()`` to create a new AudioTree with ``loudness`` calculated.
        pitch (jnp.ndarray, optional): The MIDI pitch where 60 is middle C. The shape is ``(Batch,)``.
        velocity (jnp.ndarray, optional): The MIDI velocity between 0 and 127. The shape is ``(Batch,)``.
        note_duration (jnp.ndarray, optional): A note duration in units of your choice.
            The value is not necessarily the same as the duration of the audio data. The shape is ``(Batch,)``.
        codes (jnp.ndarray, optional): The neural audio codec tokens for the audio.
        latents (jnp.ndarray, optional): The latent representations of the audio.
        metadata (dict): Any extra metadata can be placed here.
        filepaths (Union[str, Path, List[Union[str, Path]]] | None): List of filepaths for the batch of audio.

    Example:
        >>> audio = AudioTree.create(jnp.zeros((2, 44100)), 44100)  # stereo, 1 s
        >>> audio.waveform.shape
        (1, 2, 44100)
        >>> audio.sample_rate
        44100

    Note:
        If new fields are added to this class, update ``_AUDIOTREE_FIELDS`` in ``audiotree/writer.py``.
    """

    waveform: np.ndarray
    sample_rate: int = struct.field(pytree_node=False)
    loudness: np.ndarray = None
    pitch: np.ndarray = None
    velocity: np.ndarray = None
    note_duration: np.ndarray = None
    codes: np.ndarray = None
    latents: np.ndarray = None
    metadata: dict = struct.field(pytree_node=True, default_factory=dict)

    @classmethod
    def create(
        cls,
        waveform: np.ndarray,
        sample_rate: int,
        loudness: np.ndarray = None,
        pitch: np.ndarray = None,
        velocity: np.ndarray = None,
        note_duration: np.ndarray = None,
        codes: np.ndarray = None,
        latents: np.ndarray = None,
        metadata: dict = None,
        filepaths: Union[str, Path, List[Union[str, Path]]] | None = None,
    ) -> Self:
        """Create an ``AudioTree``, normalizing the waveform to ``(Batch, Channels, Samples)``.

        A bare ``(Samples,)`` or ``(Channels, Samples)`` waveform gains the missing leading axes, so
        you don't have to reshape by hand. ``filepaths`` are encoded into ``metadata``.

        Args:
            waveform: Audio of shape ``(Samples)``, ``(Channels, Samples)``, or
                ``(Batch, Channels, Samples)``.
            sample_rate: Sample rate of ``waveform`` in Hz (e.g. 44100).
            loudness: Optional precomputed LUFS loudness; usually left ``None`` and filled by
                :meth:`replace_loudness`.
            pitch: Optional MIDI pitch ``(Batch,)`` (60 = middle C).
            velocity: Optional MIDI velocity ``(Batch,)`` in ``[0, 127]``.
            note_duration: Optional per-note duration ``(Batch,)`` (not the audio duration).
            codes: Optional neural-codec tokens.
            latents: Optional latent representations.
            metadata: Optional extra metadata dict (copied, not mutated).
            filepaths: Optional path(s) for the batch; encoded into ``metadata["filepath"]``.

        Returns:
            AudioTree: A new ``AudioTree`` whose waveform is ``(Batch, Channels, Samples)``.

        Example:
            >>> audio = AudioTree.create(jnp.zeros((44100,)), 44100)  # 1 s mono
            >>> audio.waveform.shape
            (1, 1, 44100)
            >>> audio.sample_rate
            44100
        """
        # Handle audio dimensionality - ensure it's (Batch, Channels, Samples)
        if waveform.ndim == 1:
            waveform = waveform[None, None, :]  # Add batch and channel dimension
        elif waveform.ndim == 2:
            waveform = waveform[None, :, :]  # Add batch dimension

        # Handle metadata and filepaths
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()  # Don't modify the original dict
        
        if filepaths is not None:
            metadata["filepath"] = cls._encode_filepaths(filepaths)

        return cls(
            waveform=waveform,
            sample_rate=sample_rate,
            loudness=loudness,
            pitch=pitch,
            velocity=velocity,
            note_duration=note_duration,
            codes=codes,
            latents=latents,
            metadata=metadata,
        )

    @staticmethod
    def _encode_string(s: str) -> np.ndarray:
        """Encode a single filepath *s* to an array of Unicode code points.

        The returned array is shaped ``(1, _str_max_length)`` so that multiple
        rows (filepaths) can be concatenated along *axis=0*.
        """
        s = str(s)
        encoded = [ord(char) for char in s[:_str_max_length]]
        encoded += [0] * (_str_max_length - len(encoded))
        return np.array([encoded], dtype=np.int32)  # [1, _str_max_length]

    @classmethod
    def _encode_filepaths(
        cls, paths: Union[str, Path, List[Union[str, Path]]]
    ) -> np.ndarray:
        """Vectorized helper to encode one or more *paths*.

        Args:
            paths: A single filepath or an iterable of filepaths.

        Returns
        -------
        np.ndarray
            An array shaped ``(N, _str_max_length)`` where *N* is the number of
            paths provided.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        encoded_rows = [cls._encode_string(p)[0] for p in paths]
        return np.stack(encoded_rows, axis=0)

    @staticmethod
    def _decode_string(encoded_array: np.ndarray) -> str:
        """Decode a single encoded filepath back to *str*."""
        decoded = "".join(chr(int(val)) for val in encoded_array if val != 0)
        return decoded

    @property
    def filepath(self) -> List[str]:
        """Return the decoded filepaths stored in ``metadata['filepath']``.

        An empty list is returned if the AudioTree does not contain any filepath
        metadata.
        """
        if "filepath" not in self.metadata:
            return []
        return [self._decode_string(data) for data in self.metadata["filepath"]]

    @property
    def source(self) -> List[str]:
        """Return the decoded source names stored in ``metadata['source']``.

        Source names indicate which data source group each item in the batch came from.
        For example, if an AudioDataSimpleSource was created with
        ``sources={"music": [...], "speech": [...]}``, this property might return
        ``["music", "music", "speech", "music"]`` for a batch of 4 items.

        An empty list is returned if the AudioTree does not contain any source
        metadata.
        """
        if "source" not in self.metadata:
            return []
        return [self._decode_string(data) for data in self.metadata["source"]]

    @property
    def samples(self) -> int:
        """Return the number of samples in the ``waveform`` (its last dimension)."""
        return self.waveform.shape[-1]

    @property
    def batch_size(self) -> int:
        """Return the size of the leading (batch) axis.

        Derived from ``waveform``, falling back to ``codes`` / ``latents`` for
        audio-less trees (e.g. token-only training examples).
        """
        for value in (self.waveform, self.codes, self.latents):
            if value is not None:
                return value.shape[0]
        raise ValueError(
            "AudioTree has no waveform, codes, or latents to infer a batch size from."
        )

    @property
    def num_channels(self) -> int:
        """Return the number of audio channels (``waveform.shape[-2]``)."""
        return self.waveform.shape[-2]

    def __len__(self) -> int:
        """Number of items in the batch (the leading axis).

        Together with ``__getitem__`` this makes an AudioTree iterable over
        its batch items, e.g. ``for item in tree: ...`` — each ``item`` is a
        batch-of-1 AudioTree.
        """
        return self.batch_size

    def __getitem__(self, key: Union[int, slice]) -> Self:
        """Index the batch axis, returning an AudioTree of the selected item(s).

        An integer key selects a single item but keeps the leading batch axis
        (a batch of 1); a slice selects a sub-batch. Every array field —
        including ``codes``, ``latents``, and the ``metadata`` arrays — is
        indexed along the same axis so the fields stay rank-aligned.
        """
        if isinstance(key, int):
            n = self.batch_size
            if key < -n or key >= n:
                # Required for the sequence-iteration protocol: `for item in
                # tree` calls __getitem__(0), (1), ... and stops only on
                # IndexError (it does NOT consult __len__).
                raise IndexError(
                    f"batch index {key} out of range for batch_size {n}"
                )
            # Use a length-1 slice rather than a scalar index so the batch
            # axis survives on every field.
            key = slice(key, key + 1 or None)

        def _is_string_list(x) -> bool:
            return (
                isinstance(x, list) and bool(x)
                and all(isinstance(s, str) for s in x)
            )

        def _index(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)) or _is_string_list(x):
                return x[key]
            return x

        return tree_util.tree_map(_index, self, is_leaf=_is_string_list)

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the batch axis, yielding a batch-of-1 AudioTree each.

        This makes AudioTree a proper ``collections.abc.Iterable`` (the
        sequence protocol via ``__getitem__`` already allowed ``for`` loops,
        but ``isinstance(tree, Iterable)`` was ``False`` without ``__iter__``).
        Each yielded item keeps the leading batch axis, e.g. iterating a
        batch-16 tree yields 16 trees of ``batch_size == 1``.
        """
        for i in range(self.batch_size):
            yield self[i]

    def to_mono(self, strategy: Literal["average", "left", "right"] = "average") -> Self:
        """Reduce the ``waveform`` to mono.

        Args:
            strategy: ``"average"`` mixes all channels down (default);
                ``"left"`` / ``"right"`` select the corresponding channel of a
                stereo waveform.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        waveform = self.waveform
        C = self.num_channels
        if C == 1:
            return self
        if strategy == "average":
            waveform = waveform.mean(axis=-2, keepdims=True)
        elif strategy in ("left", "right") and C == 2:
            idx = 0 if strategy == "left" else 1
            waveform = waveform[..., idx:idx + 1, :]
        else:
            raise ValueError(
                f"Unsupported to_mono strategy {strategy!r} for {C} channels."
            )
        return self.replace(waveform=waveform, loudness=None)

    def to_stereo(self) -> Self:
        """Make the ``waveform`` stereo.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        waveform = self.waveform
        C = self.num_channels
        if C == 1:
            numpy = np if isinstance(waveform, np.ndarray) else jnp
            waveform = numpy.concatenate([waveform, waveform], axis=-2)
            # Duplicating the channel changes the integrated loudness (BS.1770
            # sums per-channel energy), so the cached value is no longer valid.
            return self.replace(waveform=waveform, loudness=None)
        elif C == 2:
            return self
        else:
            raise ValueError(f"Cannot make AudioTree stereo if it has {C} channels.")

    def write(
        self,
        filepath: Union[str, Path],
        subtype: str | None = None,
        format: str | None = None,
        endian: str | None = None,
    ) -> Path:
        """Write the ``waveform`` to an audio file using ``soundfile``.

        This is the inverse of :meth:`from_file`. The AudioTree must contain a
        single item (``batch_size == 1``); index or iterate the batch first
        (e.g. ``tree[0]`` or ``for item in tree``) to write each item. The
        sample rate is taken from ``self.sample_rate`` — call :meth:`resample`
        beforehand if you want a different one.

        Args:
            filepath: Output path. The file format is inferred from the
                extension (e.g. ``.wav``, ``.flac``, ``.ogg``) unless overridden
                by ``format``.
            subtype: soundfile subtype, e.g. ``"PCM_16"``, ``"PCM_24"``,
                ``"FLOAT"``. When ``None`` (default) soundfile picks the format
                default (``PCM_16`` for WAV).
            format: Major format override (e.g. ``"WAV"``, ``"FLAC"``). When
                ``None`` it is inferred from the filepath extension.
            endian: Endianness override (e.g. ``"FILE"``, ``"LITTLE"``,
                ``"BIG"``).

        Returns:
            Path: The path that was written.
        """
        assert self.batch_size == 1, (
            f"AudioTree.write requires batch_size == 1, got {self.batch_size}. "
            f"Index or iterate the batch first (e.g. tree[0])."
        )
        filepath = Path(filepath)
        # soundfile expects (samples, channels); waveform is (1, channels, samples).
        audio = np.asarray(self.waveform[0].T)
        soundfile.write(
            str(filepath),
            audio,
            self.sample_rate,
            subtype=subtype,
            format=format,
            endian=endian,
        )
        return filepath

    def resample(
        self,
        sample_rate: int,
        zeros: int = 24,
        rolloff: float = 0.945,
        output_length: int = None,
        full: bool = False,
    ) -> Self:
        """
        Resample the AudioTree's ``waveform`` to a new sample rate. The algorithm is a JAX port of ``ResampleFrac``
        from the PyTorch library `Julius`_.

        .. _Julius: https://github.com/adefossez/julius/blob/main/julius/resample.py

        Args:
            sample_rate (int): The new sample rate of audio data, such as 44100 Hz.
            zeros (int, optional): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is ``rolloff * sample_rate / 2``,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce antialiasing, but will reduce some of the
                highest frequencies.
            output_length (None or int): This can be set to the desired output length (last dimension).
                Allowed values are between 0 and ``ceil(length * sample_rate / old_sr)``. When ``None`` (default) is
                specified, the floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the ``output_length`` only
                for the last one, while passing ``full=True`` to all the other ones.

        Returns:
            AudioTree: A new ``AudioTree`` resampled to ``sample_rate`` (the original is unchanged).

        Example:
            >>> audio = AudioTree.create(jnp.zeros((44100,)), 44100)  # 1 s at 44.1 kHz
            >>> resampled = audio.resample(22050)
            >>> resampled.waveform.shape
            (1, 1, 22050)
            >>> resampled.sample_rate
            22050
        """
        if sample_rate == self.sample_rate:
            return self
        # The resample kernel is strictly 3-D, so flatten any leading axes
        # (e.g. after reshape_mini_batches) and restore them afterwards.
        leading_shape = self.waveform.shape[:-2]
        flat = self.waveform.reshape(-1, *self.waveform.shape[-2:])
        waveform = resample(
            flat,
            self.sample_rate,
            sample_rate,
            zeros=zeros,
            rolloff=rolloff,
            output_length=output_length,
            full=full,
        )
        waveform = waveform.reshape(*leading_shape, *waveform.shape[-2:])
        return self.replace(
            waveform=waveform, sample_rate=sample_rate, loudness=None
        )

    def split(self, n_splits: int) -> List[Self]:
        """Split batch dimension into a list of smaller AudioTree objects.

        Divides the batch dimension evenly into n_splits separate AudioTree objects,
        each containing a portion of the original batch.

        Args:
            n_splits: Number of AudioTree objects to create. The batch size must be
                evenly divisible by this value.

        Returns:
            List of AudioTree objects, each with batch_size = original_batch_size / n_splits.

        Example:
            >>> big_tree = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> big_tree.waveform.shape
            (12, 1, 44100)
            >>> split_trees = big_tree.split(2)
            >>> len(split_trees)
            2
            >>> split_trees[0].waveform.shape  # each tree has half the original batch size
            (6, 1, 44100)
        """
        total_batch_size = self.waveform.shape[0]
        assert total_batch_size % n_splits == 0, \
            f"Total batch size {total_batch_size} must be divisible by number of splits {n_splits}"

        split_batch_size = total_batch_size // n_splits

        return [
            tree_util.tree_map(lambda x: x[i * split_batch_size:(i + 1) * split_batch_size], self)
            for i in range(n_splits)
        ]

    def reshape_mini_batches(self, mini_batch_size: int) -> Self:
        """Reshape batch dimension into mini-batches by adding a new leading axis.

        Transforms audio data from shape (B, C, T) to (num_mini_batches, mini_batch_size, C, T),
        where B must be evenly divisible by mini_batch_size.

        Args:
            mini_batch_size: Number of samples per mini-batch. The total batch size must be
                evenly divisible by this value.

        Returns:
            AudioTree with an additional mini-batch dimension as the first axis.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.reshape_mini_batches(3)
            >>> x_batched.waveform.shape  # 4 mini-batches of size 3
            (4, 3, 1, 44100)
        """
        B = self.waveform.shape[0]

        # Calculate number of mini-batches (assuming B is evenly divisible)
        assert B % mini_batch_size == 0
        num_mini_batches = B // mini_batch_size

        # Reshape AudioTree to have leading mini-batch dimension
        # From (B, C, T) to (num_mini_batches, mini_batch_size, C, T)
        # Only reshape array-like objects since metadata can contain non-arrays
        reshaped_audio_tree = tree_util.tree_map(
            lambda x: x.reshape(num_mini_batches, mini_batch_size, *x.shape[1:]) if hasattr(x, "shape") else x,
            self,
        )
        return reshaped_audio_tree

    def flatten_mini_batches(self) -> Self:
        """Flatten mini-batches back into a single batch dimension.

        Undoes the operation performed by reshape_mini_batches(), transforming
        audio data from shape (num_mini_batches, mini_batch_size, C, T) back to
        (B, C, T).

        Returns:
            AudioTree with the mini-batch dimension flattened into the batch dimension.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.reshape_mini_batches(3)
            >>> x_batched.waveform.shape  # 4 mini-batches of size 3
            (4, 3, 1, 44100)
            >>> x_unbatched = x_batched.flatten_mini_batches()
            >>> x_unbatched.waveform.shape  # back to original shape
            (12, 1, 44100)
        """
        # Assuming the waveform has shape (num_mini_batches, mini_batch_size, C, T)
        # We want to reshape to (num_mini_batches * mini_batch_size, C, T)

        # Get the current shape
        shape = self.waveform.shape

        # We expect at least 4 dimensions for mini-batched data
        assert len(shape) >= 4, (
            f"Expected at least 4 dimensions for mini-batched data, got {len(shape)}. "
            f"Shape: {shape}"
        )

        # Flatten the first two dimensions
        # From (num_mini_batches, mini_batch_size, C, T) to (B, C, T)
        # Only reshape array-like objects since metadata can contain non-arrays
        flattened_audio_tree = tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]) if hasattr(x, "shape") else x,
            self,
        )
        return flattened_audio_tree

    def filter(self, filter_fn: Callable) -> Self:
        B = self.waveform.shape[0]
        audio_trees = self.split(B)
        audio_trees = list(filter(filter_fn, audio_trees))

        numpy = np if isinstance(self.waveform, np.ndarray) else jnp

        if len(audio_trees) == 0:
            return tree_util.tree_map(
                lambda x: x[:0] if hasattr(x, 'shape') else x,
                self
            )

        audio_trees = tree_util.tree_map(
            lambda *xs: numpy.concatenate(xs, axis=0),
            *audio_trees
        )
        return audio_trees

    @staticmethod
    def batch(items: Sequence[Any]) -> Any:
        """Batch function for use with grain's IterDataset.batch().

        Concatenates AudioTree objects along the batch axis (axis 0).
        Use this instead of grain's default batching, which would add
        an extra dimension since AudioTree already has shape (batch, channels, samples).

        Supports arbitrary nested structures containing AudioTrees. All arrays
        (including AudioTrees) are concatenated along axis 0, so data should have
        a leading batch dimension.

        Args:
            items: Sequence of AudioTree objects, or structures (dicts, lists, etc.)
                containing AudioTree objects.

        Returns:
            Batched structure with the same shape as the input items.

        Example:
            >>> a = AudioTree.create(jnp.zeros((1, 1, 16000)), 16000)
            >>> batched = AudioTree.batch([a, a, a])  # concatenate along the batch axis
            >>> batched.waveform.shape
            (3, 1, 16000)

            With Grain, pass it as the ``batch_fn`` (each item already has a leading batch axis)::

                ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)
        """
        items = list(items)

        def batching_function(*args):
            first_arg = args[0]
            if isinstance(first_arg, AudioTree):
                return _batch_audiotrees(args)
            elif isinstance(first_arg, (np.ndarray, jnp.ndarray)):
                return np.concatenate(args, axis=0)
            else:
                return list(args)

        return tree_util.tree_map(
            batching_function,
            items[0],
            *items[1:],
            is_leaf=lambda x: isinstance(x, AudioTree),
        )


def _batch_audiotrees(audio_trees: Sequence[AudioTree]) -> AudioTree:
    """Batch a list of AudioTrees into a single AudioTree.

    Concatenates all array fields along the batch axis (axis 0) using NumPy.
    Requires all AudioTrees to have the same sample_rate and compatible shapes.

    Prefer using ``AudioTree.batch`` instead, which handles mixed-type
    structures (dicts with AudioTrees, arrays, strings, etc.).

    Args:
        audio_trees: List of AudioTree objects to batch together.

    Returns:
        Single AudioTree with all items batched along axis 0.
    """
    if not audio_trees:
        raise ValueError("Cannot batch empty list of AudioTrees")

    return tree_util.tree_map(
        lambda *xs: np.concatenate(xs, axis=0),
        *audio_trees
    )
