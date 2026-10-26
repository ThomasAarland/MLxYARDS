# Website demonstration

**[Watch the 60-second walkthrough](demo.mp4).** It shows actual browser states captured
while changing the inputs and navigating the analysis and methodology tabs. It is a
captioned sequence of screenshots from real interactions, not a continuous screen recording
or a narrated student presentation. The frames are included in `demo_frames/`.

Re-encode the optional video with `python scripts/make_demo.py` (requires ffmpeg on PATH).
The website and training pipeline do not require ffmpeg.

Run `python -m streamlit run app.py` from the repository root.

The demonstration should show actual interaction with the trained model:

1. Start at own 25, first-and-10, tied, first quarter with 10:00 remaining.
2. Switch from run to pass: the selected prediction changes while both comparisons remain visible.
3. Move field position toward the opponent's end zone: field, line to gain and predictions update.
4. Move to opponent's 1: yards to go becomes 1, demonstrating goal-line validation.
5. Change down, score and clock and observe the new expectation.
6. Open Model analysis: show candidate comparison and final test results.
7. Scroll through distributions, prediction errors and feature importance.
8. Open Method & sources: explain the temporal split and observational limitations.

For a personal narrated recording, explain that expected yardage is an average, not a
guarantee. Mention that the test R² is low and that choosing run versus pass is not a
causal comparison. A 1–3 minute recording is sufficient to demonstrate the features.

The assignment allows a demonstration video instead of a public deployment. The included
video demonstrates the software; optionally replace or supplement it with your own
continuous, narrated recording using the script above.
