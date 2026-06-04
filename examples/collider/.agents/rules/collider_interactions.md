---
trigger: always_on
description: Interaction design spec for Collider app.
---

# Collider — Interaction Design Spec

> **Maintenance:** Every time you add, change, or remove an interaction behavior in this app, update this document in the same response. Do not defer it. If anything here is invalidated by a code change, revise it immediately. Notify the user of these changes and your rationale.

> **Document Scope:** Describe what the app *should do*, NOT how it's implemented, NOT how it looks. Focus on user-facing behavior and intent. Avoid referencing variable names, data structures, or algorithmic details.

## Dragging

- Dragging a ball moves it relative to the grab point — no snapping to cursor center.
- While dragging, the cursor is `grabbing` globally, even over text inputs.
- Dragging a prompt does not select it. Dragging and selection are independent.
- Hovering over a draggable element shows a `grab` cursor.

## Ball Selection

- Clicking a prompt ball (without dragging) selects that ball and blurs any focused text input.
- Clicking the canvas background clears ball selection.
- Grabbing any element other than the currently selected ball clears ball selection.
- Pressing Delete or Backspace while a ball is selected (and no text input is focused) deletes that prompt.
- Ball selection is independent of text focus — selecting a ball does not focus its text input. Focusing text does not select a ball. But selecting a ball does blur any active text input.

## Text Editing

- Clicking directly on a prompt's text focuses it and places the caret at the click position — normal input behavior.
- Focusing a prompt's text does NOT select the prompt ball. Text focus and ball selection are independent.
- Focusing text clears any active ball selection. Ball selection and text focus are mutually exclusive.
- Pressing Enter blurs the text input.
- If a text input is blurred while blank, it reverts to the value it had when focused. A prompt label can never be empty.
- Spellcheck is disabled on prompt text inputs.

## Prompt Creation

- Double-clicking the canvas creates a new prompt at that position, with its text input auto-focused with all text highlighted.
- Clicking the Add Prompt button creates a new prompt at a random position with the same auto-focus behavior.
- New prompts do not start with their ball selected.
- New prompts cycle through a placeholder label list.

## Prompt Deletion

- Pressing Delete or Backspace while a ball is selected (and no text input is focused) deletes that prompt.
- When a prompt ball is grabbed, a trash icon appears at the bottom center after a short delay to avoid flashing on quick clicks. When released, it hides.
- The trash zone renders on the bottom layer — everything else renders on top of it.
- Dropping the prompt on the trash icon deletes it.
- When the dragged prompt is over the trash zone, the icon visually expands and turns red. The hitbox does not change size.

## Influence Lines

- A dashed line connects the listener to each prompt, with dashes animating toward the listener.
- The line's opacity and thickness both reflect the prompt's current IDW weight — closer prompts produce brighter, thicker lines.
- Dash animation speed scales with proximity — closer prompts have faster-moving dashes.
- Dashes, thickness modulation, and animation can be toggled/tuned independently.

## Ring Animations

- Each ball can emit repeating CSS-animated ripple rings.
- Listener rings converge inward (scale 1→0, fading in). Prompt rings expand outward (scale 0→1, fading out).
- Prompt ring alpha is attenuated by the prompt's IDW weight — closer prompts pulse more visibly.
- Listener and prompt rings can be toggled independently.
- Ring spawning and dash animation pause when the music is not playing.

## Volume Rings

- A filled white circle is rendered behind each prompt ball. Its radius expands and contracts with the audio output level.
- Opacity is attenuated by the prompt's IDW weight — closer prompts have more visible volume rings.
- The volume level uses a peak-hold envelope: instant attack (snaps to peaks), smooth exponential release (configurable decay rate).
- The native plugin sends a single scalar audio level at ~25 Hz. The UI smooths it at 60fps for buttery animation.

## Transport

- The app starts paused. The user must explicitly press Play.
- On load, no prompt ball is selected and no text input is focused.

## Layering

- All balls (prompts and listener) are sorted by distance to the cursor — the nearest ball renders on top and gets click priority.
- Sort order only recomputes when the cursor moves, not when ball positions change.
- Sort order freezes while dragging to prevent layer shuffling mid-gesture.
- Rendering order (bottom to top): ring animations, influence lines, volume rings, balls (distance-sorted), text labels.

## Physics / Throwing

- Releasing a ball with velocity throws it. Balls bounce off container walls with no friction.
- Grabbing a moving ball stops it immediately.
- We haven't decided whether balls collide with each other. Right now it is off by default. If it is on, balls collide with each other elastically — no energy loss, equal mass.
- A ball being dragged acts as an immovable wall — other balls bounce off it.
- Moving balls should be easy to catch — use an enlarged invisible hitbox with a radius proportional to speed.
- When the window shrinks, balls clamp back into the visible bounds.
- Throws should always feel snappy — the initial launch speed matches the gesture regardless of the speed slider. The ball then gradually settles into the slider's ambient speed.

## Speed Slider

- Hidden until the first ball is thrown, then appears in the toolbar.
- Controls physics simulation speed. Full left = frozen, full right = max speed.
- Uses an exponential curve so more of the slider range is dedicated to slow speeds.
- While dragging the slider thumb, the cursor is `grabbing` globally (same treatment as balls).
