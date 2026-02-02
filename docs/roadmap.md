# Framewright Development Roadmap

This document outlines the path from the current "Cinematic Brainstorming" POC to a fully functional AI Video Editing Assistant.

## Phase 1: Core Analysis & Metadata (Current)
- [x] **Local Inference**: Self-hosted vLLM with Qwen2-VL for privacy and cost control.
- [x] **Client-Side Extraction**: Robust frame extraction ensuring video compatibility.
- [x] **Cinematic Brainstorming**: Prompt engineering for A/B-roll, camera info, and transitions.
- [x] **Structured Data**: Storing Clip Metadata in DuckDB.
- [ ] **External API Integration**: Full support for Gemini/OpenAI video analysis models for comparison/fallback.

## Phase 2: Search & Retrieval (Next)
- [ ] **Vector Search**: Implement embedding-based search (e.g., "Find me a happy shot at the beach") using CLIP/SigLIP.
- [ ] **Semantic Tagging**: Auto-tagging clips with specific objects, moods, and colors.
- [ ] **Natural Language Querying**: "Show me all drone shots of the city at night."

## Phase 3: The "Paper Edit" (Assembly)
- [ ] **Timeline Generation**: Convert search results + narrative prompt into an EDL (Edit Decision List) or XML timeline.
- [ ] **Rough Cut Export**: Export `.xml` or `.otio` files compatible with Premiere/Resolve.
- [ ] **Music Sync**: Basic beat detection to suggest cutting points that align with audio.

## Phase 4: Full Assistant (Conversational)
- [ ] **Interactive Chat**: "Replace the second clip with something more energetic."
- [ ] **Context Awareness**: The AI understands the current state of the timeline.
- [ ] **Multi-Modal Feedback**: The user can scrub video usage and the AI updates its plan.

## Phase 5: Production Ready
- [ ] **Proxy Workflow**: Analyzing low-res proxies but linking to high-res source files.
- [ ] **Cloud Sync**: Optional cloud storage for team collaboration.
- [ ] **Plugin Integration**: Direct plugin for NLEs (Premiere/Resolve) to bypass export/import steps.
