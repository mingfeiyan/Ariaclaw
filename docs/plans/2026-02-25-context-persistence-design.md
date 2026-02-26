# AriaClaw Context Persistence Design

**Date**: 2026-02-25
**Status**: Approved

## Goal

Persist multimodal context from Aria Gen 2 glasses (vision, audio, spatial, biometric) so the AI assistant builds useful memory over time — within sessions, across sessions, and long-term.

## Design Principles

- **Capture transitions, not states** — log when things change, not continuous streams
- **Derived context over raw data** — store structured events, not raw sensor dumps
- **Write Markdown to OpenClaw** — leverage OpenClaw's memory system for indexing and retrieval
- **Local-first processing** — all processing on Mac + Aria on-device, minimal cloud dependency
- **Always-on** — context capture runs independently of active AI conversation

---

## Sensor Configuration

### Always-On (low cost, high value)

| Sensor | Config | Rationale |
|---|---|---|
| Microphones (8-mic array) | Always on, 16kHz mono | Speech is highest-value context |
| Contact microphone | Always on | Distinguishes wearer's voice from others |
| SLAM (on-device) | On-device processing, stream position events only | Aria computes on-chip |
| Eye tracking (on-device) | On-device processing, stream gaze events only | Aria computes on-chip |

### Adaptive (change-driven)

| Sensor | Config | Rationale |
|---|---|---|
| RGB Camera | Idle: 1 frame/10s. Active: 1 fps on scene change. Burst: 3 fps on new scene. | Most frames are redundant |
| IMU | Log orientation changes > 30deg and significant movement | Head turns indicate context shifts |

### Periodic/Background

| Sensor | Config | Rationale |
|---|---|---|
| PPG (heart rate) | Sample every 30s. Log if HR deviates > 20% from baseline | Emotional context marker |
| GNSS | Every 2 min moving, every 10 min stationary | Geo-tagging context |
| Barometer/Magnetometer | Every minute | Floor level, compass heading |

### Scene Change Detection

- Compare current frame to last keyframe using perceptual hash (pHash)
- Similarity < 80% (hamming distance > 10) -> scene changed -> capture at 1fps for 5 seconds
- Similarity > 95% for 30 seconds -> drop to idle rate (1 frame/10s)

---

## Memory Architecture (Three Tiers)

### Tier 1: Sensory Buffer (seconds)

Circular buffer, constantly overwritten. Allows context pipeline to reach back for higher-fidelity data when something interesting is detected.

| Data | Retention | Size |
|---|---|---|
| Raw audio stream | 30 seconds rolling | ~960KB |
| Raw RGB frames | Last 10 frames | ~5MB |
| Raw sensor readings | 30 seconds rolling | ~100KB |

### Tier 2: Short-Term Memory (hours, up to 24h)

Structured context events written as Markdown to OpenClaw daily logs (`memory/YYYY-MM-DD.md`).

**Context Event types:**

| Type | Trigger | Example |
|---|---|---|
| `speech_self` | Contact mic + transcription | "I said: let's reschedule to Thursday" |
| `speech_other` | Mic array + transcription | "They said: the deadline is Friday" |
| `scene_change` | RGB frame diff | "Entered conference room, 4 people at table" |
| `gaze_focus` | Eye tracking sustained > 2s | "Looked at laptop screen showing email" |
| `location_change` | SLAM/GNSS transition | "Moved from desk to kitchen" |
| `activity_change` | IMU + scene combined | "Sat down" / "Started walking" |
| `heart_rate_spike` | PPG deviation > 20% | "Elevated HR: 95bpm (baseline 68)" |
| `object_detected` | RGB frame analysis | "New object: coffee cup, notebook" |
| `text_detected` | OCR on RGB frame | "Sign reads: Building 40, Room 201" |

**Daily log format:**

```markdown
# 2026-02-25

## 09:15 - Meeting (Conference Room B, 4 people)
- **You said**: "Let's move the deadline to Thursday"
- **Other said**: "That works, but we need the design doc by Wednesday"
- **Decision**: Deadline moved to Thursday, design doc due Wednesday
- **Scene**: [keyframe](context/2026-02-25/keyframe-0915.jpg)
- **Mood**: Engaged, slightly elevated HR (82 bpm, baseline 68)

## 09:47 - Walking to desk
- **Location**: Building 40, 2nd floor
```

**Storage**: ~100-300MB per full day (Markdown + keyframe JPEGs).

### Tier 3: Long-Term Memory (days, indefinite)

AI-compressed summaries written to OpenClaw's `MEMORY.md`.

**Day summary structure** (appended to MEMORY.md):

- Time-ranged segments (meeting, commute, work, break)
- Per segment: location, people, summary, key facts, representative keyframes
- Extracted: decisions, commitments, people encountered, places visited

**Forgetting curve:**

| Age | What remains |
|---|---|
| Day 1-7 | Full Tier 2 daily logs |
| Day 8-30 | Pruned daily logs (only events referenced by summaries + starred) |
| Day 30+ | Only Tier 3 summaries + keyframes in MEMORY.md |
| Never deleted | Decisions, commitments, people, starred events |

**Storage**: ~1-5MB per day. Full year = ~500MB-2GB.

---

## OpenClaw Integration

Our pipeline writes directly to OpenClaw's filesystem-based memory:

| Our output | OpenClaw target | How it's used |
|---|---|---|
| Tier 2 context events | `~/.openclaw/workspace/memory/YYYY-MM-DD.md` | Indexed by file watchers, searchable via `memory_search` |
| Tier 3 summaries | `~/.openclaw/workspace/MEMORY.md` | Loaded at session start, never decays |
| Keyframe JPEGs | `context/YYYY-MM-DD/` (referenced by path in Markdown) | Retrieved via `memory_get` when visual context needed |
| Session transcripts | `~/.openclaw/agents/<agentId>/sessions/*.jsonl` | Optional session memory indexing |

OpenClaw provides semantic search (vector + BM25), temporal decay, and MMR re-ranking over our Markdown files automatically.

---

## Processing Pipeline

### New component: `context_pipeline.py`

```
ariaclaw.py (coordinator)
├── aria_stream.py           # existing
├── gemini_service.py        # existing
├── audio_playback.py        # existing
├── openclaw_bridge.py       # existing
├── dashboard.py             # existing
└── context_pipeline.py      # NEW
    ├── audio_processor.py   # transcription + speaker separation
    ├── scene_processor.py   # keyframe capture + description
    ├── spatial_processor.py # SLAM/IMU/GPS event filtering
    └── memory_writer.py     # writes structured Markdown to OpenClaw
```

### Stream 1: Audio -> Conversation Context

```
Mic array (16kHz) -> Contact mic separation -> Whisper transcription -> Context events
```

- Speaker separation via contact mic (physical, not ML)
- Transcription via Whisper (local, `small` or `medium` model, Metal-accelerated)
- Sentence grouping via punctuation + silence gaps
- Prosody analysis via pitch tracking + RMS energy (scipy, not ML)

### Stream 2: Video -> Scene Context

```
RGB frames -> Scene change detection (pHash) -> Keyframe saved -> Scene description -> Context events
```

- Scene change detection via perceptual hash (imagehash library)
- Keyframe storage to `context/YYYY-MM-DD/keyframe-HHMMSS.jpg`
- Scene description via Gemini Flash (batched every 5 min) or local VLM fallback

### Stream 3: Spatial + Biometric -> Environment Context

All classical signal processing, no ML:

| Signal | Algorithm | Library |
|---|---|---|
| PPG -> Heart rate | Bandpass filter (0.7-3.5 Hz) + peak detection | scipy.signal |
| HR anomaly | Exponential moving average + z-score (> 2 sigma) | numpy |
| Activity classification | Accelerometer variance + gravity vector angle | numpy |
| Location transition | SLAM position delta > 3m or room change | numpy |
| Gaze target | Sustained fixation > 2s, map to keyframe region | numpy |

### Summarization Pipeline (end of day)

1. Read today's `memory/YYYY-MM-DD.md`
2. AI (Gemini Flash) extracts: key conversations, decisions, commitments, people, places
3. Append summary to `MEMORY.md`
4. Prune old daily logs per forgetting curve

---

## Models & Algorithms

### ML Models (v1)

| Model | Task | Runs on | Size | Cost |
|---|---|---|---|---|
| Whisper `small` | Speech transcription | Mac (whisper.cpp + Metal) | 500MB | Free |
| Gemini Flash | Scene description + daily summarization | Cloud (batched) | N/A | ~$0.01-0.05/day |

### Signal Processing (no ML)

| Task | Algorithm | Library |
|---|---|---|
| PPG -> HR | Bandpass filter + peak detection | scipy.signal |
| HR baseline & anomaly | EMA + z-score | numpy |
| Audio prosody | Pitch (autocorrelation) + RMS + speech rate | scipy/librosa |
| Activity classification | Accelerometer variance + gravity angle | numpy |
| Scene change | Perceptual hash + hamming distance | imagehash |
| OCR | Apple Vision framework | pyobjc + Vision |

### Future (v2+)

| Model | Task | When needed |
|---|---|---|
| LLaVA 7B | Offline scene description fallback | If offline capability needed |
| nomic-embed-text | Local semantic search | If OpenClaw cloud embeddings undesirable |

### Resource Budget (M-series Mac, 16GB+ RAM)

- RAM: ~1.7GB (Whisper + signal processing)
- GPU: Moderate (Metal for Whisper)
- Network: Minimal (batched Gemini calls)
- Cost: ~$0.01-0.05/day

---

## Context Retrieval & Query Flow

### At Query Time

```
User speaks -> Gemini receives question
  1. Check today's daily log (already loaded by OpenClaw)
  2. If not found -> memory_search (semantic search over all indexed logs)
  3. If visual context needed -> retrieve keyframe JPEG
  4. Gemini answers with assembled context
```

### Real-Time Context Injection (configurable)

| Trigger | Injection |
|---|---|
| New conversation detected | Last 2 min of transcription |
| Scene change | Latest scene description |
| Location change | Place name + time |

Default: inject scene changes and conversation transcripts only.

### Context Window Budget

- Live context (last 5 min transcription + current scene): ~500 tokens
- Session context (today + yesterday daily logs): ~2-4K tokens
- Retrieved context (memory_search on demand): ~500-1K tokens per query
- Total overhead: ~3-5K tokens baseline

---

## Gemini System Instruction Addition

```
You have access to the wearer's context memory.
When they ask about past events, people, places, or conversations:
1. Check today's context (already in your session)
2. Use memory_search to find relevant past context
3. If a keyframe is referenced, use memory_get to retrieve it

Context entries include: conversations (who said what),
scene descriptions, locations, gaze focus, and biometric hints.
```
