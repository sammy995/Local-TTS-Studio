# Architecture Documentation

## Overview

This project implements **Hexagonal Architecture** (Ports & Adapters) with strict layer separation. The architecture supports **dual runtime** (local + cloud) from a single codebase without tech debt.

## Core Principle

> **Local and Cloud are environments, not products.**  
> Build ONE core system, TWO runtimes.

## Layer Structure

```
TTS-opensource-app/
├── core/              # Pure compute (no I/O, no FastAPI, no environment)
│   ├── model_manager.py    # Model loading & inference
│   ├── tts_engine.py       # Compute orchestration
│   └── audio_pipeline.py   # Data transformations
│
├── services/          # Business orchestration (stateless)
│   └── tts_service.py      # Use case workflows
│
├── infra/             # Side effects only (storage, queue, auth)
│   └── storage.py          # Protocol-based storage abstraction
│
└── runtimes/          # Thin entrypoints (< 600 lines each)
    ├── local_api.py        # Local FastAPI server
    └── config_loader.py    # Runtime configuration
```

---

## Layer Responsibilities

### 🔵 `core/` — Pure Compute

**Allowed:**
- torch, numpy, transformers
- Pure data transformation
- Synchronous functions only

**Forbidden:**
- File I/O (no `with open()`, no `Path.write_bytes()`)
- FastAPI imports
- Environment awareness (`if local:`, `if cloud:`)
- Hardcoded paths

**Returns:**
- Simple types: `numpy.ndarray`, `bytes`, `dict`, `str`
- Engine returns `(audio_array, sample_rate)` tuples (NOT bytes)

**Purity Test:**
```python
from core.tts_engine import TTSEngine
from core.model_manager import ModelManager

# Must work in plain Python REPL
manager = ModelManager(...)
engine = TTSEngine(manager)
audio_array, sr = engine.generate_custom_voice("hello", "speaker_1")
# ✅ No file I/O, no FastAPI, just compute
```

---

### 🟡 `services/` — Orchestration

**Allowed:**
- Calling engine
- Validation logic
- Coordinating workflows
- Calling storage **interface** (passed per-call)

**Forbidden:**
- Storage as instance variable (must be stateless)
- Direct file I/O
- Environment branching (`if local:`)
- FastAPI types in signatures

**Design Pattern:**
```python
# ❌ WRONG: Storage in constructor (stateful)
class TTSService:
    def __init__(self, engine, storage):
        self.storage = storage  # ❌ Couples environment

# ✅ CORRECT: Storage per-call (stateless)
class TTSService:
    def __init__(self, engine):
        self.engine = engine  # ✅ Pure

    def generate_speech(self, text, storage):  # ✅ Storage passed
        audio_array, sr = self.engine.generate(...)
        bytes = to_wav_bytes(audio_array, sr)
        return storage.save_audio(bytes, ...)
```

**Think of services as pure functions:**
```python
generate_speech(request, storage) → file_id
```

---

### 🟠 `infra/` — Side Effects

**Allowed:**
- Disk I/O
- S3 uploads
- Database calls
- Redis caching
- Auth logic

**Forbidden:**
- Business logic
- Generation logic
- Request validation

**Storage Interface (Protocol):**
```python
class Storage(Protocol):
    def save_audio(self, audio_bytes: bytes, filename: str) -> str: ...
    def save_prompt(self, tensor: torch.Tensor, name: str) -> str: ...
    def load_prompt(self, name: str) -> torch.Tensor: ...
```

**Implementations:**
- `LocalStorage`: Saves to `./outputs/`
- `CloudStorage`: Saves to S3 (future)

No inheritance required — structural typing via Protocol.

---

### 🔴 `runtimes/` — Entrypoints

**Allowed:**
- FastAPI routes
- Request parsing
- Progress tracking (SSE)
- Dependency composition

**Forbidden:**
- Generation logic
- Storage logic
- Business validation

**Composition Pattern:**
```python
# Initialize dependencies
storage = LocalStorage(...)
manager = ModelManager(...)
engine = TTSEngine(manager)
service = TTSService(engine)

# Routes just compose
@app.post("/api/tts/custom-voice")
async def generate_custom_voice(...):
    # 1. Parse request
    # 2. Call service.generate_speech(storage=storage, ...)
    # 3. Return response
```

**Target:** < 600 lines per runtime file.

---

## Architecture Rules

### ✅ DO

1. **Engine returns arrays, pipeline encodes**
   ```python
   audio_array, sr = engine.generate(...)  # Returns numpy
   bytes = pipeline.to_wav_bytes(audio_array, sr)  # Encodes to WAV
   ```

2. **Services stay stateless**
   ```python
   service = TTSService(engine)  # No storage
   result = service.generate_speech(..., storage=storage)  # Pass per-call
   ```

3. **Storage via Protocol (structural typing)**
   ```python
   class Storage(Protocol):  # Not ABC
       def save_audio(...): ...
   ```

4. **Inject all parameters**
   ```python
   # ❌ WRONG
   manager = ModelManager()  # Reads hardcoded paths

   # ✅ CORRECT
   manager = ModelManager(
       model_base_path=Path("./models"),
       device="cuda",
       dtype=torch.bfloat16
   )
   ```

### ❌ DON'T

1. **No environment branching in services**
   ```python
   # ❌ FORBIDDEN
   if local:
       storage.save_local()
   else:
       storage.save_s3()
   
   # ✅ CORRECT
   storage.save_audio(...)  # Polymorphic
   ```

2. **No FastAPI in core/services**
   ```python
   # ❌ FORBIDDEN
   from fastapi import HTTPException
   
   # ✅ CORRECT
   raise ValueError("...")  # Use stdlib
   ```

3. **No file I/O in core**
   ```python
   # ❌ FORBIDDEN
   with open("output.wav", "wb") as f:
       f.write(audio_data)
   
   # ✅ CORRECT
   return audio_bytes  # Caller handles I/O
   ```

4. **No async in core/services**
   ```python
   # ❌ FORBIDDEN
   async def generate(...):
   
   # ✅ CORRECT
   def generate(...):  # Sync only
   ```

---

## Validation

### Purity Tests

Run `python test_architecture.py` to validate:

1. ✅ Core has no forbidden imports (FastAPI, boto3)
2. ✅ Core works in Python REPL (zero environment setup)
3. ✅ Engine returns `(array, sample_rate)` tuples
4. ✅ Service is stateless (storage passed per-call)
5. ✅ Storage uses Protocol (structural typing)
6. ✅ Service has no `if local:` branches
7. ✅ Runtime is thin (< 600 lines)

**Expected Output:**
```
🎉 ALL TESTS PASSED (7/7)
✅ Architecture is clean and production-ready
```

---

## Benefits

### 1. **Dual Runtime Support**
- Same core/services for local + cloud
- Just swap `LocalStorage` → `CloudStorage`
- Zero logic duplication

### 2. **Easy Testing**
```python
# Mock storage for tests
class DummyStorage:
    def save_audio(self, bytes, filename):
        return f"dummy_{filename}"

service = TTSService(engine)
result = service.generate_speech(..., storage=DummyStorage())
```

### 3. **Reusable Core**
```python
# Use in Jupyter notebook
from core.tts_engine import TTSEngine
engine = TTSEngine(...)
audio, sr = engine.generate("hello", "speaker_1")

# Use in CLI script
audio_bytes = to_wav_bytes(audio, sr)
Path("output.wav").write_bytes(audio_bytes)
```

### 4. **Future-Proof**
- Swap models: Just change `ModelManager`
- Add S3: Implement `CloudStorage`
- Add batch workers: Reuse `service.generate_batch()`
- Publish as library: `pip install tts-core`

---

## Migration Path (Completed ✅)

### Phase 1: Core Extraction ✅
- [x] Extract `core/audio_pipeline.py` (pure transformations)
- [x] Clean `core/model_manager.py` (parameter injection)
- [x] Create `core/tts_engine.py` (returns arrays)

### Phase 2: Service Layer ✅
- [x] Build `services/tts_service.py` (stateless)
- [x] Storage passed per-call

### Phase 3: Infrastructure ✅
- [x] Implement `infra/storage.py` (Protocol)
- [x] `LocalStorage` implementation

### Phase 4: Runtime ✅
- [x] Refactor to `runtimes/local_api.py` (thin wrapper)
- [x] Move `config_loader.py` to runtime
- [x] All tests passing

---

## What's Next

### Cloud Runtime (v2)
```python
# runtimes/cloud_api.py
storage = CloudStorage(bucket="tts-outputs")  # S3
manager = ModelManager(...)  # Same core
engine = TTSEngine(manager)  # Same core
service = TTSService(engine)  # Same service

# Different: auth, queue, CloudStorage
@app.post("/api/tts/custom-voice")
@requires_auth  # Cloud-specific
async def generate_custom_voice(...):
    job_id = service.generate_speech(storage=storage, ...)
    return {"job_id": job_id}
```

95% code shared. Only runtime plumbing differs.

---

## Anti-Patterns to Avoid

### ❌ Frankenstein Product
```python
# DON'T fork logic
if environment == "local":
    result = generate_local(...)
else:
    result = generate_cloud(...)
```

### ❌ Leaky Abstraction
```python
# DON'T leak environment into service
class TTSService:
    def generate(self, text):
        if self.is_local:  # ❌ Violates agnosticism
            save_local()
```

### ❌ Stateful Service
```python
# DON'T store storage
class TTSService:
    def __init__(self, engine, storage):
        self.storage = storage  # ❌ Couples environment
```

---

## Quotes from Refactoring

> "Local and Cloud are environments, not products. Build ONE core system, TWO runtimes."

> "Get it slightly wrong → tech debt forever."

> "Most devs say 'we'll refactor later'. They never do."

> "Abstractions are debt until proven useful."

> "Senior engineers delete abstractions aggressively."

---

## Summary

This architecture is:
- ✅ **Minimal** (no speculative interfaces)
- ✅ **Pragmatic** (solves today's pain, enables tomorrow's features)
- ✅ **Testable** (purity tests enforce boundaries)
- ✅ **Production-ready** (passes all validation)

Direction: **Bulletproof.**
