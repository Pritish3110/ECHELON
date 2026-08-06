# ECHELON — Multilingual Voice-Controlled Agentic Assistant

## Project Guide

## 1. Core Idea
A voice-driven agentic system: speak a command (Hindi/English) → ASR transcribes → LangGraph agent decides the right tool → RAG / web scrape / (stretch) phone control → response synthesized → TTS speaks it back.

**Primary narrative for resume/interviews:** "I built a multilingual voice agent that routes commands to specialized tools (retrieval, web research, device control) using an agentic orchestration layer, deployed on AWS free-tier infra with production-style guardrails."

---

## 2. Architecture

```
Voice Input
    │
    ▼
ASR (dots.tts stack, multilingual Hindi/English)
    │
    ▼
LangGraph Orchestrator Agent (routing + state)
    │
    ├── RAG Tool ──────► OpenSearch (vector + BM25 hybrid)
    ├── Firecrawl Tool ─► Web scrape/research → feeds RAG ingestion
    └── [STRETCH] Mobile Control Tool ─► ADB + Gemini 2.0 Flash (vision)
    │
    ▼
Response Synthesis (LLM)
    │
    ▼
TTS (dots.tts) → Voice Output
```

**Backend wiring:** API Gateway → Lambda (agent backend) → Cognito (auth) → DynamoDB (state/logs) → CloudTrail (audit).

---

## 3. Tech Stack (Finalized)

| Layer | Tech | Notes |
|---|---|---|
| ASR | dots.tts / fine-tuned multilingual model | Your existing MiraiMinds work; focus on low latency |
| TTS | dots.tts | Hindi-English |
| Agent orchestration | LangGraph | Routing, state machine between tools |
| Agent tooling | LangChain | Tool wrappers, chains |
| Tool exposure | MCP Server | Standardized tool interface (RAG, Firecrawl, mobile-control) |
| Prompting | Prompt engineering | Routing prompts, tool-selection, guardrail prompts |
| Vision/reasoning LLM | Gemini 2.0 Flash (free tier) | Used for mobile screen control decisions |
| Text LLM | Groq or local Ollama (RTX 3060) | RAG synthesis, general reasoning |
| Retrieval | RAG pipeline (chunking + embeddings) | Custom or LangChain-native |
| Vector DB + search | AWS OpenSearch (free tier) | Hybrid BM25 + kNN — replaces separate Elasticsearch + vector DB |
| Scraping | Firecrawl | 500 credits/month free — feeds RAG ingestion |
| Mobile control | ADB + Gemini Vision | **Stretch goal**, timeboxed |
| Auth | AWS Cognito | Google as federated IdP — replaces separate GAuth integration |
| Permissions | AWS IAM | Least-privilege roles per Lambda function |
| API layer | AWS API Gateway | Rate limiting via usage plans |
| Compute | AWS Lambda | Serverless agent backend |
| Audit | AWS CloudTrail | First trail free forever |
| Database | AWS DynamoDB | Replaces Firestore — avoids cross-cloud auth overhead |
| Guardrails | Custom middleware | Input: prompt-injection detection. Output: hallucination/PII filter |

**Dropped/merged from original brainstorm:**
- Elasticsearch + Vector DB → merged into OpenSearch
- Cognito + Google Auth → merged (Cognito federates Google)
- Firestore → replaced by DynamoDB (swap back only with a specific reason)

---

## 4. Free-Tier Reality Check

| Service | Free tier limit | Duration |
|---|---|---|
| Lambda | 1M requests/month | Forever |
| API Gateway | 1M calls/month | 12 months only |
| Cognito | 50k MAUs | Forever |
| CloudTrail | 1 trail | Forever |
| DynamoDB | 25GB + 25 WCU/RCU | Forever |
| OpenSearch | Single t3.small.search node | **12 months only** |
| Gemini 2.0 Flash | ~15 RPM / 1500 req/day (verify current limits) | Ongoing, rate-limited |
| Firecrawl | 500 scrape credits/month | Ongoing |

**Non-negotiable:** Set an AWS Billing Alert (~$5 threshold) before writing any infra code. New AWS account needed for a fresh 12-month clock if you've used free tier before.

---

## 5. Build Phases (Risk-Ordered, Not Feature-Ordered)

### Phase 1 — Core Voice Loop (MVP)
- ASR → simple LLM call → TTS, no agent logic yet.
- Goal: prove low-latency speech round-trip works.
- **Exit criteria:** sub-X-second round trip, works in Hindi + English.

### Phase 2 — RAG + OpenSearch
- Ingest a small doc set → OpenSearch hybrid search → LLM synthesis.
- Goal: working retrieval, isolated from voice layer.
- **Exit criteria:** accurate answers on test queries, standalone (text-in/text-out).

### Phase 3 — Agent Orchestration (LangGraph)
- Wire ASR/TTS + RAG into a LangGraph agent with tool routing.
- Add Firecrawl as a second tool.
- **Exit criteria:** agent correctly routes between RAG vs. scrape based on query type.

### Phase 4 — AWS Infra + Auth
- Deploy backend on Lambda + API Gateway, add Cognito auth, IAM roles, CloudTrail.
- Add rate limiting (API Gateway usage plans).
- **Exit criteria:** end-to-end request goes through auth + rate limiting + Lambda, not just local.

### Phase 5 — Guardrails
- Input: prompt-injection detection layer.
- Output: hallucination/PII filter before TTS.
- **Exit criteria:** demonstrable block of at least one injection attempt and one unsafe output in testing.

### Phase 6 — MCP Server Exposure
- Wrap your tools (RAG, Firecrawl) behind an MCP server interface.
- **Exit criteria:** tools callable from an external MCP client (e.g., Claude Desktop) as a demo.

### Phase 7 — [STRETCH, TIMEBOXED] Mobile Control
- ADB + Gemini Vision screen-reading loop, exposed as an agent tool.
- **Hard timebox: 1-2 weeks.** If not demo-stable by then, cut it — ship without it.
- **Exit criteria:** agent can execute at least 3 distinct phone actions reliably (e.g., open app, send message, search).

---

## 6. Key Risks (Flagged Honestly)

1. **Solo scope** — this spans speech ML, cloud infra, agent orchestration, and mobile automation. Normally 4-5 specialized roles.
2. **"AI agents handle everything" mindset** — agents can scaffold, but you must own debugging in routing logic, latency, and security-adjacent code.
3. **Mobile vision-control is unreliable by nature** — known failure rates in this category (AppAgent, Mobile-Agent, etc.). Treat as research, not integration.
4. **Free-tier is a countdown clock**, not a permanent safety net — OpenSearch (12mo) and API Gateway (12mo) especially.
5. **Integration debt compounds silently** — auth → Lambda → OpenSearch → agent → ADB is a long chain; test each boundary in isolation before connecting.
6. **Scope creep** — this idea already grew 3x in early planning. Lock scope per phase; don't add mid-build.

---

## 7. Git Branch Management

Use a single repo, feature-branch-per-phase, PR-to-merge workflow. This matches your phased build directly and keeps `main` always in a working, demo-able state.

**Branch structure:**
```
main                          → always stable, demo-ready
├── feature/voice-loop        → Phase 1
├── feature/rag-opensearch    → Phase 2
├── feature/agent-orchestration → Phase 3
├── feature/aws-infra-auth    → Phase 4
├── feature/guardrails        → Phase 5
├── feature/mcp-server        → Phase 6
└── feature/mobile-control    → Phase 7 (stretch, timeboxed)
```

**Workflow:**
1. Branch off `main` per phase: `git checkout -b feature/rag-opensearch`
2. Commit incrementally within the phase (don't wait till the whole phase is done to commit).
3. Open a PR back into `main` only when that phase's exit criteria (Section 5) are met.
4. Self-review the PR diff before merging — this is where you catch integration issues early, not after everything's wired together.
5. Merge → tag if it's a milestone (e.g., `v0.1-core-voice-loop`) → delete the feature branch.
6. Never build two phases in the same branch — it defeats the isolation that's protecting you from integration debt (Section 6, risk #5).

**Why this matters for you specifically:** since agents will scaffold a lot of code, PRs are your checkpoint to actually read and understand what got generated before it merges into main — don't skip the review step even if it "looks fine."

---

## 9. Immediate Next Actions
- [ ] Set AWS Billing Alert
- [ ] Confirm current Gemini 2.0 Flash free-tier rate limits (verify — changes periodically)
- [ ] Start Phase 1 (core voice loop) — no AWS, no agent, just ASR→LLM→TTS
- [ ] Only move to Phase 2 once Phase 1 exit criteria are met
