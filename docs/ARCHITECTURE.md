# Architecture and technical decisions

## Scope

**Use case:** local engineering handbook Q&A, with exact source inspection when the chat model calls a tool. This lab is designed to be understandable in one session. No general agent orchestration, hosted backend, or shared production data.

**Why this is different from TracePilot:** TracePilot already covers incident investigations, runbook retrieval, multi-role reviews and observability in a Java/Spring stack. This repo is a Python/LangChain reference implementation for the two essential mechanisms: *retrieval of grounded snippets* and *model-selected, validated, read-only tool calls*. Do not port TracePilot's incident domain or multi-agent workflow here.

## Data and execution flow

1. On CLI startup, `KnowledgeBase` enumerates up to 100 local `.md` files. It skips symlinked files, limits each file to 200 KB, decodes UTF-8 and stamps a relative source name.
2. LangChain `RecursiveCharacterTextSplitter` produces 800-character chunks with 120-character overlap, maintaining source/chunk metadata.
3. `TokenHashEmbeddings` (offline lexical baseline) or `OpenAIEmbeddings` produces vectors, indexed in `InMemoryVectorStore` for cosine similarity search. Nothing is persisted.
4. The query retrieves up to 3 chunks with similarity >= 0.12. Snippets are passed to a guarded system prompt as `[S1]`, `[S2]`, `[S3]` with filename and chunk number.
5. `ChatOpenAI.bind_tools` enables just `list_documents` and `read_document`; the response's structured `tool_calls` are examined in the application, not dynamically executed.
6. Each requested tool name is compared with a fixed allowlist. `read_document` arguments are validated by Pydantic and the filename must match an indexed source exactly. Tool output is bounded (1,600 characters for an excerpt, 2,000 for a tool message); two tool executions maximum.
7. If the model returns an answer, referenced `[S*]` labels must exist in the original retrieval context. This catches impossible labels but **does not prove** that the model used the cited sentence correctly. The CLI prints source files and provider token counts separately from the answer.

## Error boundaries

- Missing/invalid docs, embedding configuration, invalid question, and missing API key yield user-facing errors and nonzero exit codes.
- Provider calls have client-side timeout and retries. Internal provider exception strings are suppressed from CLI output to reduce the chance of accidental credential exposure.
- Invalid tool names/arguments and tool failures are returned to the model as bounded tool messages, rather than being executed or silently accepted.
- Repeated model tool requests stop after the budget; if the model still requests tools, the application emits a budget message instead of inventing a result.

## Trade-offs

| Choice | Why | Cost / limitation |
| --- | --- | --- |
| Python + LangChain Core | Small, inspectable integration surface | LangChain API/version compatibility matters |
| In-memory vector store | No external database; deterministic startup | Re-index on every run; small corpus only |
| Lexical hash default | Offline demo and test reliability | Not semantic; poor synonym matching |
| Optional OpenAI embeddings | Real semantic retrieval comparison | API cost, latency and internet required |
| Manual tool loop | Audit how structured calls are validated | More plumbing than a general agent framework |
| Short bounded excerpts | Limits tokens, accidental disclosure, and memory | May exclude critical evidence from larger docs |
| Labeled hit-at-k fixture | Easy to run in CI without model cost | Does not measure hallucinations or truthfulness |

## Operational notes

The Docker container has read-only root FS, no writable volume and no local network listener. Ingestion is local and the chat provider is reached only with explicit credentials. If the documents are confidential, use an approved provider or self-hosted model before trying live inference. Real security testing, prompt-injection evaluation, secrets scanning, semantic answer scoring and authorization are **out of scope** for this learning lab.
