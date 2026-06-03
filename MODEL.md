---
license: cc-by-4.0
library_name: magenta-realtime-2
---

# Model Card for Magenta RealTime 2

**Authors**: Google DeepMind

**Resources**:

-   [Get Started](https://magenta.withgoogle.com/mrt2)
-   [Blog Post](https://magenta.withgoogle.com/magenta-realtime-2)
-   [Repository](https://github.com/magenta/magenta-realtime)
-   [HuggingFace](https://huggingface.co/google/magenta-realtime-2)

## Terms of Use

Magenta RealTime 2 is offered under a combination of licenses: the codebase is
licensed under
[Apache 2.0](https://github.com/magenta/magenta-realtime/blob/main/LICENSE), and
the model weights under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).
In addition, we specify the following usage terms:

Copyright 2026 Google LLC

Use these materials responsibly and do not generate content, including outputs,
that infringe or violate the rights of others, including rights in copyrighted
content.

Google claims no rights in outputs you generate using Magenta RealTime 2. You
and your users are solely responsible for outputs and their subsequent uses.

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

Magenta RealTime 2 is an open music generation model from Google built for on
device streaming generation with low-latency control. It is a
[live music model](https://arxiv.org/abs/2508.04651) and a follow up to the
prior [Magenta RealTime model](https://huggingface.co/google/magenta-realtime)
and [Lyria RealTime API](http://goo.gle/lyria-realtime), offering on-device
generation with richer control and lower latency. Magenta RealTime 2 enables the
continuous generation of musical audio steered by text prompts, audio examples,
and MIDI.

### System Components

Magenta RealTime 2 is composed of three components: SpectroStream, MusicCoCa,
and an LLM. The structure is similar to that of the original Magenta RealTime,
detailed [here](https://arxiv.org/abs/2508.04651). The primary difference is
the LLM, which is now a Decoder-only model supporting frame-wise autoregression
(rather than chunk-wise) and tuned for on-device streaming with frame-level
control.

1.  **SpectroStream** ([Li+ 25](https://arxiv.org/abs/2508.05207)) is a
    discrete audio codec that converts stereo 48kHz audio into tokens.
1.  **MusicCoCa** is a contrastive-trained model capable of embedding audio and
    text into a common embedding space, building on
    [Yu+ 22](https://arxiv.org/abs/2205.01917) and
    [Huang+ 22](https://arxiv.org/abs/2208.12415).
1.  A **decoder-only Transformer LLM** generates audio tokens given context
    audio tokens, a tokenized MusicCoCa embedding, and MIDI tokens. There are
    two configurations:
      1. A `base` configuration with 2.4B parameters
      1. A `small` configuration with 230M parameters

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
-   **Decoder Transformer LLM**: Generates audio tokens given context, MIDI,
    and style. At each timestep (codec frame), the model receives:
    -   **Input**:
        - (Context) SpectroStream tokens
          - `base`: 25 frame (1s) windowed attention per layer, 20 layers
          - `small`: 41 frame (~1.6s) windowed attention per layer, 12 layers
          - Yields 20s effective receiptive field for both models
        - (Style) 12 MusicCoCa tokens
        - (MIDI) 128-dim multihot vector representing the state of each MIDI
          pitch during this frame (0 = Off, 1 = Sustain, 2 = Onset, 3 = Sustain
          or onset, model decides)
    -   **Output**: 1 generated frame, 12 RVQ tokens

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
        MusicCoCa and Magenta RT 2 to recognize musical information.
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

Magenta RT 2 supports the real-time generation and steering of instrumental
music. The purpose and intention of this capability is to foster the
development of new real-time, interactive co-creation workflows that seamlessly
integrate with human-centered forms of musical creativity.

Every AI music generation model, including Magenta RT 2, carries a risk of
impacting the economic and cultural landscape of music. We aim to mitigate these
risks through the following avenues:

-   Prioritizing human-AI interaction as fundamental in the design of Magenta
    RT 2.
-   Distributing the model under a terms of service that prohibit developers
    from generating outputs that infringe or violate the rights of others,
    including rights in copyrighted content.
-   Training on primarily instrumental data. With specific prompting, this model
    has been observed to generate some vocal sounds and effects, though those
    vocal sounds and effects tend to be non-lexical.

### Known limitations

Magenta RealTime 2 has similar limitations to Magenta RealTime in terms of
genre coverage and non lexical vocalizations,
[refer here for details](https://huggingface.co/google/magenta-realtime#known-limitations).

### Benefits

At the time of release, Magenta RealTime 2 represents the only open weights
model supporting real-time, continuous musical audio generation with low
latency control (~200ms). It is designed specifically to enable live,
interactive musical creation, bringing new capabilities to musical
performances, art installations, video games, and many other applications.

## How to Get Started with the Model

See our [Get Started Page](https://magenta.withgoogle.com/mrt2)
and [GitHub repository](https://github.com/magenta/magenta-realtime) for usage
examples.

## Training Details

### Training Data

Magenta RealTime 2 was trained on ~71k hours of stock music from multiple
sources, mostly instrumental.

### Hardware

Magenta RealTime 2 was trained using
[Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu)
hardware.

### Software

Training was done using [JAX](https://github.com/jax-ml/jax) and
[Sequence Layers](https://github.com/google/sequence-layers). JAX allows
researchers to take advantage of the latest generation of hardware, including
TPUs, for faster and more efficient training of large models.

## Evaluation

Model evaluation metrics and results will be shared in our forthcoming technical
report.

## Citation

A paper about Magenta RealTime 2 is forthcoming. For now, please cite our
previous technical report:

**BibTeX:**

```
@inproceedings{gdmlyria2025live,
    title={Live Music Models},
    author={Caillon, Antoine and McWilliams, Brian and Tarakajian, Cassie and Simon, Ian and Manco, Ilaria and Engel, Jesse and Constant, Noah and Li, Pen and Denk, Timo I. and Lalama, Alberto and Agostinelli, Andrea and Huang, Anna and Manilow, Ethan and Brower, George and Erdogan, Hakan and Lei, Heidi and Rolnick, Itai and Grishchenko, Ivan and Orsini, Manu and Kastelic, Matej and Zuluaga, Mauricio and Verzetti, Mauro and Dooley, Michael and Skopek, Ondrej and Ferrer, Rafael and Borsos, Zal{\'a}n and van den Oord, {\"A}aron and Eck, Douglas and Collins, Eli and Baldridge, Jason and Hume, Tom and Donahue, Chris and Han, Kehang and Roberts, Adam},
    booktitle={NeurIPS Creative AI},
    year={2025}
}
```
