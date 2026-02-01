# Podcast Mode - Content Production Compiler

## Overview

Podcast Mode is **not** a multi-voice TTS feature. It's a **deterministic multi-speaker production pipeline** - a script-to-audio compiler for content creation.

**Strategic Positioning:**
- Not: "TTS with many voices"
- But: "Script-to-podcast compiler"
- Category: Content production infrastructure (workflows get paid, features get compared)

---

## Core Concepts

### Three Data Primitives

```
PodcastProject
  ├── Speakers (reusable voice personas)
  ├── Segments (atomic speech units)
  └── Render settings (deterministic/creative mode)
```

**Speakers** = Voice identities (like actors in a movie)  
**Segments** = Atomic compilation units (fault-tolerance boundaries)  
**Project** = Complete production definition

---

## Architecture

### Compiler Pipeline

```
Project JSON → Validate → Precompute Prompts → Render Segments → Concat → Encode → Save
```

**Key Decisions:**
1. **Segment-level atomicity** - Each segment is independent (retries, caching, progress, parallelization)
2. **Prompt caching** - Same speaker 50 times = 1 prompt generation (huge optimization)
3. **Array-only compute** - No temp files, memory-safe concatenation
4. **Deterministic by default** - Same input → identical bytes (compiler mode)
5. **Runtime controls progress** - Service stays unaware (clean separation)

---

## Data Models

### Speaker

```python
@dataclass(frozen=True)
class Speaker:
    id: str                      # Unique identifier
    name: str                    # Human-readable name
    mode: "custom" | "design" | "clone"
    config: Dict[str, Any]       # Mode-specific config (NOT validated)
    style_instruction: Optional[str]
```

**Config Keys (documented, not enforced):**
- `mode='custom'`: `{'voice_id': 'Female-1'}`
- `mode='design'`: `{'description': 'young female, cheerful'}`
- `mode='clone'`: `{'prompt_id': 'my_voice_01'}`

**Why Dict:** Future-proof for pitch/tone/accent without schema migrations

---

### Segment

```python
@dataclass(frozen=True)
class Segment:
    id: str                      # Unique identifier
    order: int                   # Sort order (allow gaps: 10, 20, 30)
    speaker_id: str              # Speaker reference
    text: str                    # Text to synthesize
    pause_after_ms: int = 500    # Silence gap after segment
    volume: float = 1.0          # Gain multiplier (0.5 = half, 2.0 = double)
    emotion: Optional[str] = None
```

**Order Strategy:**
- Allow gaps (10, 20, 30) - easy insertions without renumbering
- Only enforce uniqueness (no duplicates)
- No contiguous requirement (prevents migration headaches)

---

### PodcastProject

```python
@dataclass(frozen=True)
class PodcastProject:
    id: str
    title: str
    speakers: List[Speaker]
    segments: List[Segment]
    output_format: "wav" | "mp3" = "wav"
    target_sample_rate: int = 44100
    deterministic: bool = True   # Compiler mode (default)
```

**Deterministic Mode (default):**
- Greedy decoding (temperature=0)
- Stable hashing for seeds (sha256)
- Identical output bytes every time
- Position: "Build mode" (reliable, production-ready)

**Creative Mode (optional):**
- Stochastic sampling (temp=0.9)
- Variety across renders
- Position: "Draft mode" (experimental, expressive)

---

## Service API

### PodcastService

**Core Methods:**

```python
# Structural validation only (cheap + deterministic)
validate_project(project: PodcastProject) -> None

# Atomic segment rendering (retry unit)
render_segment(
    segment: Segment,
    speaker: Speaker,
    prompt_cache: Dict,
    deterministic: bool = True
) -> Tuple[np.ndarray, int]

# Convenience wrapper (memory-safe concat)
render_podcast(
    project: PodcastProject,
    storage: Storage,
    pipeline: AudioPipeline
) -> str
```

**Validation Rules (Service Layer):**
- Max 10 speakers
- Max 5000 chars per segment
- Max 60 min estimated duration
- Speaker references valid
- Segment order uniqueness

**Does NOT Validate:**
- Text language/chars (engine decides)
- Model availability (engine decides)
- Config keys (engine interprets)

---

## Determinism Strategy

### Greedy Decoding (Preferred)

```python
if deterministic:
    temperature = 0.0  # Strictly deterministic
    top_p = 1.0
    top_k = 1
```

**Why Greedy > Seeded Sampling:**
- Greedy = always identical (even if model changes slightly)
- Sampling + seed = may drift with model updates
- Cleaner semantics for "compiler mode"

### Stable Seeding

```python
# SHA256 hash of segment ID → stable across machines/versions
seed = hashlib.sha256(segment.id.encode()).digest()[:8]
seed_int = int.from_bytes(seed, byteorder='big')
```

**NOT** Python's `hash()` (randomized per process)

---

## Progress Tracking

**Segment-Level Granularity:**

```python
progress = (completed_segments / total_segments) * 100
```

**Ownership:**
- Runtime emits progress (after each segment)
- Service stays completely unaware
- Simple, predictable, monotonic

**NO:**
- Complex percentage formulas
- Progress inside service (would couple)
- Sub-segment granularity (noise)

---

## Memory Safety

**For Long Podcasts (60 min ≈ 600MB):**

```python
# Calculate total length first
total_length = sum(len(arr) for arr in arrays)

# Preallocate once
final_array = np.zeros(total_length, dtype=arrays[0].dtype)

# Copy segments (no repeated concat)
offset = 0
for arr in arrays:
    final_array[offset:offset + len(arr)] = arr
    offset += len(arr)
```

**Avoid:** Repeated `np.concatenate()` in loop (memory explosion)

---

## Runtime Integration

### Local (Synchronous)

```
POST /api/podcast/render
→ parse JSON
→ loop segments with progress
→ return audio file
```

**Progress:** Emit after each segment via SSE

### Cloud (Asynchronous) - Future

```
POST /api/podcast/jobs
→ create RenderJob
→ enqueue worker
→ return job_id

GET /api/podcast/jobs/{id}
→ status + progress

GET /api/podcast/jobs/{id}/download
→ audio file
```

**Worker:** Same service code, just wrapped in job queue

---

## Example Request

```json
{
  "id": "ep001",
  "title": "My First Podcast",
  "deterministic": true,
  "speakers": [
    {
      "id": "host",
      "name": "Host",
      "mode": "custom",
      "config": {"voice_id": "Female-1"},
      "style_instruction": "cheerful and engaging"
    },
    {
      "id": "guest",
      "name": "Guest",
      "mode": "design",
      "config": {"description": "deep male voice, serious"},
      "style_instruction": null
    }
  ],
  "segments": [
    {
      "id": "seg_01",
      "order": 10,
      "speaker_id": "host",
      "text": "Welcome to the show!",
      "pause_after_ms": 1000,
      "volume": 1.0
    },
    {
      "id": "seg_02",
      "order": 20,
      "speaker_id": "guest",
      "text": "Thanks for having me.",
      "pause_after_ms": 500,
      "volume": 0.9
    }
  ],
  "output_format": "wav",
  "target_sample_rate": 44100
}
```

---

## Monetization Strategy

**DO monetize:**
- ✅ Longer duration (free: 10 min, paid: 60 min)
- ✅ Batch size (free: 1 project, paid: 50 projects)
- ✅ Faster cloud rendering
- ✅ Project storage/history
- ✅ Team collaboration

**DON'T monetize:**
- ❌ Determinism (reliability feature, not premium)
- ❌ Creative mode (just a toggle, not a gate)

**Positioning:**
- Free tier: Reliable compiler mode (build confidence)
- Paid tier: Scale + speed + collaboration

---

## Benefits Over "Multi-Voice TTS"

### 1. **Fault Tolerance**
- Segment-level retries (not full restart)
- Failure in segment 47/50 = retry 1 segment

### 2. **Performance**
- Prompt caching (1 generation for 50 uses)
- Parallel segments (future)
- Incremental progress

### 3. **Determinism**
- Same script = identical audio
- Version control friendly
- A/B testing valid

### 4. **Workflow Product**
- Not a feature, infrastructure
- Batchable, automatable, scriptable
- Enterprise-friendly

### 5. **Cloud-Ready**
- Job model from day one
- Same service code for local/cloud
- No forked logic

---

## Future Extensions

### Trivial to Add
- Music underlay (volume mixing)
- Sound effects (insert at timestamps)
- Voice EQ/filters (post-processing)
- Parallel segment rendering (worker pool)
- Resume from checkpoint (partial artifacts)

### Why Trivial
- Segment atomicity preserved
- Service stays stateless
- Runtime handles orchestration

---

## Strategic Insight

This is **not TTS infrastructure**.  
This is **content production infrastructure**.

Same category as:
- Video editors (Premiere, Final Cut)
- DAWs (Logic, Ableton)
- CI/CD pipelines (GitHub Actions)

Workflows = paid subscriptions  
Features = compared and copied

You're building a **compiler**, not a **demo**.

That's the difference between toy AI app and production system.

---

## Summary

**What it is:**
- Script-to-audio compiler
- Deterministic by default
- Segment-level atomicity
- Prompt caching optimization
- Memory-safe concatenation
- Progress-aware runtime

**What it isn't:**
- Multi-voice TTS feature
- Interactive voice chat
- Real-time synthesis
- Voice playground

**Why it matters:**
- Enterprise use cases
- Automation-friendly
- Production-grade reliability
- Aligns with governance/systems career path

**Status:** Production-ready architecture ✅
