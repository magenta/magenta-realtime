---
license: cc-by-4.0
library_name: magenta-realtime
---

# Model Card for Magenta RT

**Authors**: Google DeepMind

**Resources**:

-   [Blog Post](https://g.co/magenta/rt)
-   [Paper](https://arxiv.org/abs/2508.04651)
-   [Colab Demo](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb)
-   [Repository](https://github.com/magenta/magenta-realtime)
-   [HuggingFace](https://huggingface.co/google/magenta-realtime)

## Terms of Use

Magenta RealTime is offered under a combination of licenses: the codebase is
licensed under
[Apache 2.0](https://github.com/magenta/magenta-realtime/blob/main/LICENSE), and
the model weights under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).
In addition, we specify the following usage terms:

Copyright 2025 Google LLC

Use these materials responsibly and do not generate content, including outputs,
that infringe or violate the rights of others, including rights in copyrighted
content.

Google claims no rights in outputs you generate using Magenta RealTime. You and
your users are solely responsible for outputs and their subsequent uses.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses. You are solely responsible for
determining the appropriateness of using, reproducing, modifying, performing,
displaying or distributing the software and materials, and any outputs, and
assume any and all risks associated with your use or distribution of any of the
software and materials, and any outputs, and your exercise of rights and
permissions under the licenses.

## Model Details

Magenta RealTime is an open music generation model from Google built from the
same research and technology used to create
[MusicFX DJ](https://labs.google/fx/tools/music-fx-dj) and
[Lyria RealTime](http://goo.gle/lyria-realtime). Magenta RealTime enables the
continuous generation of musical audio steered by a text prompt, an audio
example, or a weighted combination of multiple text prompts and/or audio
examples. Its relatively small size makes it possible to deploy in environments
with limited resources, including live performance settings or freely available
Colab TPUs.

### System Components

Magenta RealTime is composed of three components: SpectroStream, MusicCoCa, and
an LLM. A full technical report with more details on each component is
[here](https://arxiv.org/abs/2508.04651).

1.  **SpectroStream** is a discrete audio codec that converts stereo 48kHz audio
    into tokens, building on the SoundStream RVQ codec from
    [Zeghidour+ 21](https://arxiv.org/abs/2107.03312)
1.  **MusicCoCa** is a contrastive-trained model capable of embedding audio and
    text into a common embedding space, building on
    [Yu+ 22](https://arxiv.org/abs/2205.01917) and
    [Huang+ 22](https://arxiv.org/abs/2208.12415).
1.  An **encoder-decoder Transformer LLM** generates audio tokens given context
    audio tokens and a tokenized MusicCoCa embedding, building on the MusicLM
    method from [Agostinelli+ 23](https://arxiv.org/abs/2301.11325)

### Inputs and outputs

-   **SpectroStream RVQ codec**: Tokenizes high-fidelity music audio
    -   **Encoder input / Decoder output**: Music audio waveforms, 48kHz stereo
    -   **Encoder output / Decoder input**: Discrete audio tokens, 25Hz frame
        rate, 64 RVQ depth, 10 bit codes, 16kbps
-   **MusicCoCa**: Joint embeddings of text and music audio
    -   **Input**: Music audio waveforms, 16kHz mono, or text representation of
        music style e.g. "heavy metal"
    -   **Output**: 768 dimensional embedding, quantized to 12 RVQ depth, 10 bit
        codes
-   **Encoder-decoder Transformer LLM**: Generates audio tokens given context
    and style
    -   **Encoder Input**: (Context, 1000 tokens) 10s of audio context tokens w/
        4 RVQ depth, (Style, 6 tokens) Quantized MusicCoCa style embedding
    -   **Decoder Output**: (Generated, 800 tokens) 2s of audio w/ 16 RVQ depth

## Uses

Music generation models, in particular ones targeted for continuous real-time
generation and control, have a wide range of applications across various
industries and domains. The following list of potential uses is not
comprehensive. The purpose of this list is to provide contextual information
about the possible use-cases that the model creators considered as part of model
training and development.

-   **Interactive Music Creation**
    -   Live Performance / Improvisation: These models can be used to generate
        music in a live performance setting, controlled by performers
        manipulating style embeddings or the audio context
    -   Accessible Music-Making & Music Therapy: People with impediments to
        using traditional instruments (skill gaps, disabilities, etc.) can
        participate in communal jam sessions or solo music creation.
    -   Video Games: Developers can create a custom soundtrack for users in
        real-time based on their actions and environment.
-   **Research**
    -   Transfer learning: Researchers can leverage representations from
        MusicCoCa and Magenta RT to recognize musical information.
-   **Personalization**
    -   Musicians can finetune models with their own catalog to customize the
        model to their style (fine tuning support coming soon).
-   **Education**
    -   Exploring Genres, Instruments, and History: Natural language prompting
        enables users to quickly learn about and experiment with musical
        concepts.

### Out-of-Scope Use

See our [Terms of Use](#terms-of-use) above for usage we consider out of scope.

## Bias, Risks, and Limitations

Magenta RT supports the real-time generation and steering of instrumental music.
The purpose and intention of this capability is to foster the development of new
real-time, interactive co-creation workflows that seamlessly integrate with
human-centered forms of musical creativity.

Every AI music generation model, including Magenta RT, carries a risk of
impacting the economic and cultural landscape of music. We aim to mitigate these
risks through the following avenues:

-   Prioritizing human-AI interaction as fundamental in the design of Magenta
    RT.
-   Distributing the model under a terms of service that prohibit developers
    from generating outputs that infringe or violate the rights of others,
    including rights in copyrighted content.
-   Training on primarily instrumental data. With specific prompting, this model
    has been observed to generate some vocal sounds and effects, though those
    vocal sounds and effects tend to be non-lexical.

### Known limitations

**Coverage of broad musical styles**. Magenta RT's training data primarily
consists of Western instrumental music. As a consequence, Magenta RT has
incomplete coverage of both vocal performance and the broader landscape of rich
musical traditions worldwide. For real-time generation with broader style
coverage, we refer users to our
[Lyria RealTime API](g.co/magenta/lyria-realtime).

**Vocals**. While the model is capable of generating non-lexical vocalizations
and humming, it is not conditioned on lyrics and is unlikely to generate actual
words. However, there remains some risk of generating explicit or
culturally-insensitive lyrical content.

**Latency**. Because the Magenta RT LLM operates on two second chunks, user
inputs for the style prompt may take two or more seconds to influence the
musical output.

**Limited context**. Because the Magenta RT encoder has a maximum audio context
window of ten seconds, the model is unable to directly reference music that has
been output earlier than that. While the context is sufficient to enable the
model to create melodies, rhythms, and chord progressions, the model is not
capable of automatically creating longer-term song structures.

### Benefits

At the time of release, Magenta RealTime represents the only open weights model
supporting real-time, continuous musical audio generation. It is designed
specifically to enable live, interactive musical creation, bringing new
capabilities to musical performances, art installations, video games, and many
other applications.

## How to Get Started with the Model

See our
[Colab demo](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb)
and [GitHub repository](https://github.com/magenta/magenta-realtime) for usage
examples.

## Training Details

### Training Data

Magenta RealTime was trained on ~190k hours of stock music from multiple
sources, mostly instrumental.

### Hardware

Magenta RealTime was trained using
[Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu)
hardware (TPUv6e / Trillium).

### Software

Training was done using [JAX](https://github.com/jax-ml/jax) and
[T5X](https://github.com/google-research/t5x), utilizing
[SeqIO](https://github.com/google/seqio) for data pipelines. JAX allows
researchers to take advantage of the latest generation of hardware, including
TPUs, for faster and more efficient training of large models.

## Evaluation

Model evaluation metrics and results will be shared in our forthcoming technical
report.

## Citation

Please cite our technical report:

**BibTeX:**

```
@article{gdmlyria2025live,
    title={Live Music Models},
    author={Caillon, Antoine and McWilliams, Brian and Tarakajian, Cassie and Simon, Ian and Manco, Ilaria and Engel, Jesse and Constant, Noah and Li, Pen and Denk, Timo I. and Lalama, Alberto and Agostinelli, Andrea and Huang, Anna and Manilow, Ethan and Brower, George and Erdogan, Hakan and Lei, Heidi and Rolnick, Itai and Grishchenko, Ivan and Orsini, Manu and Kastelic, Matej and Zuluaga, Mauricio and Verzetti, Mauro and Dooley, Michael and Skopek, Ondrej and Ferrer, Rafael and Borsos, Zal{\'a}n and van den Oord, {\"A}aron and Eck, Douglas and Collins, Eli and Baldridge, Jason and Hume, Tom and Donahue, Chris and Han, Kehang and Roberts, Adam},
    journal={arXiv:2508.04651},
    year={2025}
}
```
