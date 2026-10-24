# Idea review: learn from YouTube ASL videos, and Nemotron 3 Ultra via OpenRouter

Date: 2026-10-24. Status: critique only, nothing built. Phase 1 step 4/6 are still open; this is later-phase work (Phase 2 for phrases).

## Idea A: extract landmarks from tutorial/daily-use ASL videos and train on them

What is sound: the real bottleneck is signer diversity. The baseline overfits 15 training signers
(train 0.85–0.99 vs test 0.46–0.48; signer 29302 at 0.25–0.27). More signers is the right lever.

| # | Loophole | Fix |
|---|---|---|
| 1 | I cannot "watch" and learn. A video skill gives frames; MediaPipe gives landmarks; learning happens only by training a model on them. Any sign meaning I assign by looking at frames is unverified and can be wrong (rule 7). | Treat it as a data-extraction pipeline feeding `ml/` training, never as Claude's judgement. Labels come from the video's own text, checked by a fluent signer. |
| 2 | Labels. Tutorials give one gloss per clip at best; daily-use videos give English captions, not glosses, with no timing per sign. Captions are translations, so word-for-word alignment is false. | Isolated signs: only clips with an explicit gloss and a visible citation form. Phrases need CTC (Phase 2) and a gloss layer. Do not mix. |
| 3 | Licence and consent (rules 2, 4; ADR-0003). YouTube's terms restrict downloading; creators hold copyright; Deaf creators did not consent to model training. | Research track only, and only for videos with a CC licence or written permission. Never into a release model or bundle. Check YouTube-ASL (Google, ~1,000 h, 2,500+ signers per its paper) licence with `/dataset-license` first; its terms were not confirmed here `[VERIFY]`. For release: Studio recordings under consent form, or Deaf creators who license CC BY. |
| 4 | Domain gap. Tutorial signers are often hearing learners, slow, exaggerated, mirrored, edited, compressed; camera angle, fps (24/30/60) and hand z differ (z already `[VERIFY]` as ~2x smaller in browser clips). | Measure first: run the current model zero-shot on 5 signs × a few videos. Resample fps, mirror-check, drop non-fluent signers (advisor review). |
| 5 | Evaluation leakage. If your test signing resembles training videos, accuracy looks better than it is. | Keep signer-independent test sets: Kaggle test signers plus your own recordings, never in training. |
| 6 | Speed variation is real but already handled cheaply: time-warp/speed augmentation exists or is trivial; it does not need video. | Add speed augmentation before scraping anything. |
| 7 | Storage/privacy. Videos are identifiable people. | Process locally, keep only landmarks, delete video; treat landmarks as personal data. |
| 8 | Phase and cost. Adds a scraping + alignment project before the G0 vocabulary exists. | Minimum now: a one-day zero-shot domain-gap probe on a licensed handful. Everything else waits for G0 and Phase 2. |

## Idea B: Nemotron 3 Ultra through OpenRouter

Facts (OpenRouter listing, 2026-10-24): 550B total / 55B active MoE, hybrid Transformer-Mamba, **text in, text out only**, released 2026-06-04, paid $0.50 in / $2.20 out per 1M tokens plus a free trial endpoint. Model licence not confirmed `[VERIFY]`; context length listings disagree (512K vs 1M).

Verdict: not worth it.
- It cannot see video or landmarks, so it does not help recognition. The recognizer is a small on-device model.
- Runtime use breaks rules 1 and 2 (API key, paid, data leaves the device).
- Dev-time use (gloss normalisation, caption cleanup) is marginal and the free endpoint runs under trial terms that may log prompts; no Deaf-user data may be sent.
- Revisit only for Phase 4 gloss↔English translation experiments, with a locally runnable open model first.

## Recommended next step

Zero-shot domain-gap probe (item 4), research track, local only, after `/dataset-license`. Otherwise stay on the plan: G0 vocabulary, advisor outreach, own Studio recordings for signer diversity.

Sources: [OpenRouter listing](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b), [YouTube-ASL paper](https://arxiv.org/abs/2306.15162v2).
