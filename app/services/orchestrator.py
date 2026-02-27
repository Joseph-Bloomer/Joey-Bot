"""Chat pipeline orchestrator — 7-stage pipeline for message processing."""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Generator

from app.prompts import get_mode_prefix
from app.data.database import Conversation, Message
from app.services.web_search import WebSearchService
from utils.logger import get_logger
import config

logger = get_logger()


class ChatOrchestrator:
    """
    Orchestrates the chat pipeline through 7 sequential stages:
      1. CLASSIFY    — gatekeeper decides if memory retrieval is needed
      2. WEB_SEARCH  — optional Tavily web search (based on mode + classification)
      3. RETRIEVE    — hybrid dense + BM25 search
      4. SCORE       — heuristic reranker (or passthrough if disabled)
      5. BUILD_CONTEXT — assemble the LLM prompt
      6. GENERATE    — stream tokens via SSE
      7. POST_PROCESS — update access metadata, extract facts
    """

    def __init__(self, local_llm, model_registry, gatekeeper, retriever,
                 chat_service, memory_service, reranker=None, compressor=None):
        self.local_llm = local_llm
        self.model_registry = model_registry  # {display_name: BaseLLM instance}
        self.gatekeeper = gatekeeper
        self.retriever = retriever
        self.chat_service = chat_service
        self.memory_service = memory_service
        self.reranker = reranker      # HeuristicReranker instance (or None)
        self.compressor = compressor  # Future placeholder
        self.web_search = WebSearchService()
        self._last_pipeline_ctx: Optional[Dict[str, Any]] = None

    def process_message(
        self,
        user_message: str,
        conversation_id: Optional[int],
        recent_messages: List[Dict[str, str]],
        mode: str = "normal",
        search_mode: str = "auto",
        model: str = None,
    ) -> Generator[str, None, None]:
        """
        Run the full pipeline and yield SSE-formatted tokens.

        Args:
            user_message: Current user message text.
            conversation_id: Saved conversation ID (or None).
            recent_messages: Pre-loaded recent message dicts.
            mode: Response mode (normal/concise/logic).
            search_mode: Web search mode — "off", "auto", or "on".
            model: Display name of model to use for generation (resolved via registry).

        Yields:
            SSE data strings: {"token": "..."}, {"searching": true}, and {"done": true}.
        """
        ctx: Dict[str, Any] = {
            "user_message": user_message,
            "conversation_id": conversation_id,
            "recent_messages": recent_messages,
            "mode": mode,
            "search_mode": search_mode,
            "model": model,
            "classification": {},
            "candidates": [],
            "scored_candidates": [],
            "web_results": None,
            "context_parts": {},
            "assembled_prompt": "",
            "response_text": "",
            "timings": {},
            "errors": [],
        }

        # Stages 1-4 run synchronously before streaming
        self._stage_classify(ctx)
        yield from self._stage_web_search(ctx)
        self._stage_retrieve(ctx)
        self._stage_score(ctx)
        self._stage_build_context(ctx)

        # Stage 5 is a generator — yield from it
        yield from self._stage_generate(ctx)

        # Stage 6 runs after streaming completes
        self._stage_post_process(ctx)

        # Log pipeline summary
        self._log_pipeline_summary(ctx)
        self._last_pipeline_ctx = ctx

    # ------------------------------------------------------------------
    # Stage 1: CLASSIFY
    # ------------------------------------------------------------------

    def _stage_classify(self, ctx: Dict[str, Any]) -> None:
        t0 = time.perf_counter()
        try:
            if self.gatekeeper and config.GATEKEEPER_ENABLED:
                gatekeeper_context = ctx["recent_messages"][-config.GATEKEEPER_CONTEXT_WINDOW:] if ctx["recent_messages"] else []
                classification = self.gatekeeper.classify(ctx["user_message"], gatekeeper_context)
            else:
                classification = {
                    "memory_need": "SEMANTIC",
                    "retrieval_keys": [],
                    "confidence": 0.0,
                }
            ctx["classification"] = classification
        except Exception as e:
            ctx["classification"] = {
                "memory_need": "SEMANTIC",
                "retrieval_keys": [],
                "confidence": 0.0,
            }
            ctx["errors"].append({"stage": "CLASSIFY", "error": str(e), "fallback": "SEMANTIC"})
            logger.warning(f"[CLASSIFY] Error: {e}, defaulting to SEMANTIC")

        elapsed = (time.perf_counter() - t0) * 1000
        ctx["timings"]["classify"] = elapsed
        c = ctx["classification"]
        logger.info(
            f"[CLASSIFY] {c['memory_need']} "
            f"(conf={c.get('confidence', 0):.2f}, {elapsed:.1f}ms)"
        )

    # ------------------------------------------------------------------
    # Stage 1.5: WEB_SEARCH
    # ------------------------------------------------------------------

    def _stage_web_search(self, ctx: Dict[str, Any]) -> Generator[str, None, None]:
        """Execute web search if gatekeeper or user toggle requests it."""
        t0 = time.perf_counter()
        search_mode = ctx.get("search_mode", "auto")
        memory_need = ctx["classification"].get("memory_need", "SEMANTIC")
        retrieval_keys = ctx["classification"].get("retrieval_keys", [])

        # Determine whether to search
        should_search = False
        if search_mode == "off":
            pass
        elif search_mode == "on":
            should_search = True
        elif search_mode == "auto":
            should_search = (memory_need == "WEB_SEARCH")

        if not should_search:
            elapsed = (time.perf_counter() - t0) * 1000
            ctx["timings"]["web_search"] = elapsed
            logger.info(f"[WEB_SEARCH] skipped (mode={search_mode}, memory_need={memory_need}, {elapsed:.1f}ms)")
            return

        if not self.web_search.is_available:
            elapsed = (time.perf_counter() - t0) * 1000
            ctx["timings"]["web_search"] = elapsed
            logger.warning(f"[WEB_SEARCH] unavailable (missing API key or disabled, {elapsed:.1f}ms)")
            return

        # Signal frontend that a search is starting
        yield f'data: {json.dumps({"searching": True})}\n\n'

        try:
            query = " ".join(retrieval_keys) if retrieval_keys else ctx["user_message"]
            web_results = self.web_search.search(query)
            ctx["web_results"] = web_results

            elapsed = (time.perf_counter() - t0) * 1000
            ctx["timings"]["web_search"] = elapsed

            result_count = web_results.get("result_count", 0)
            error = web_results.get("error")
            if error:
                logger.warning(f'[WEB_SEARCH] query="{query}" error={error} ({elapsed:.1f}ms)')
            else:
                logger.info(f'[WEB_SEARCH] query="{query}" results={result_count} ({elapsed:.1f}ms)')

        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            ctx["timings"]["web_search"] = elapsed
            ctx["web_results"] = None
            ctx["errors"].append({"stage": "WEB_SEARCH", "error": str(e), "fallback": "no web results"})
            logger.warning(f"[WEB_SEARCH] Error: {e} ({elapsed:.1f}ms)")

    # ------------------------------------------------------------------
    # Stage 2: RETRIEVE
    # ------------------------------------------------------------------

    def _stage_retrieve(self, ctx: Dict[str, Any]) -> None:
        t0 = time.perf_counter()
        memory_need = ctx["classification"].get("memory_need", "SEMANTIC")

        if memory_need not in ("SEMANTIC", "MULTI"):
            ctx["candidates"] = []
            elapsed = (time.perf_counter() - t0) * 1000
            ctx["timings"]["retrieve"] = elapsed
            logger.info(f"[RETRIEVE] Skipped (memory_need={memory_need}, {elapsed:.1f}ms)")
            return

        try:
            if self.memory_service.store.count() == 0:
                ctx["candidates"] = []
                elapsed = (time.perf_counter() - t0) * 1000
                ctx["timings"]["retrieve"] = elapsed
                logger.info(f"[RETRIEVE] 0 candidates — store empty ({elapsed:.1f}ms)")
                return

            retrieval_keys = ctx["classification"].get("retrieval_keys", [])
            query = " ".join(retrieval_keys) if retrieval_keys else ctx["user_message"]

            query_embedding = self.memory_service.llm.get_embedding(query)
            if query_embedding is None:
                ctx["candidates"] = []
                elapsed = (time.perf_counter() - t0) * 1000
                ctx["timings"]["retrieve"] = elapsed
                logger.warning(f"[RETRIEVE] Embedding returned None ({elapsed:.1f}ms)")
                return

            # Append extra keywords for BM25
            bm25_query = query
            if retrieval_keys:
                bm25_query = ctx["user_message"] + " " + " ".join(retrieval_keys)

            candidates = self.retriever.search(
                query_text=bm25_query,
                query_embedding=query_embedding,
                n_results=config.SEMANTIC_RESULTS_COUNT,
            )
            ctx["candidates"] = candidates

        except Exception as e:
            ctx["candidates"] = []
            ctx["errors"].append({"stage": "RETRIEVE", "error": str(e), "fallback": "empty candidates"})
            logger.warning(f"[RETRIEVE] Error: {e}")

        elapsed = (time.perf_counter() - t0) * 1000
        ctx["timings"]["retrieve"] = elapsed
        logger.info(f"[RETRIEVE] {len(ctx['candidates'])} candidates ({elapsed:.1f}ms)")

    # ------------------------------------------------------------------
    # Stage 3: SCORE (heuristic reranker)
    # ------------------------------------------------------------------

    def _stage_score(self, ctx: Dict[str, Any]) -> None:
        t0 = time.perf_counter()
        candidates = ctx["candidates"]

        if not candidates:
            ctx["scored_candidates"] = []
            elapsed = (time.perf_counter() - t0) * 1000
            ctx["timings"]["score"] = elapsed
            logger.info(f"[SCORE] no candidates to score ({elapsed:.1f}ms)")
            return

        if self.reranker is not None and config.RERANKER_ENABLED:
            try:
                before_count = len(candidates)
                scored = self.reranker.rerank(
                    candidates,
                    query_context={},
                    top_k=config.SEMANTIC_RESULTS_COUNT,
                )
                after_count = len(scored)
                ctx["scored_candidates"] = scored

                # Log per-candidate breakdown
                for c in scored:
                    bd = c.get("score_breakdown", {})
                    logger.info(
                        f"[SCORE]   {c['text'][:50]}... "
                        f"composite={bd.get('composite', 0):.3f} "
                        f"(rel={bd.get('retrieval_relevance', 0):.2f} "
                        f"rec={bd.get('recency', 0):.2f} "
                        f"imp={bd.get('importance', 0):.2f} "
                        f"use={bd.get('usage', 0):.2f} "
                        f"type={bd.get('type_boost', 0):.2f})"
                    )

                dropped = before_count - after_count
                if dropped > 0:
                    logger.info(f"[SCORE] dropped {dropped} low-scoring candidates")

                elapsed = (time.perf_counter() - t0) * 1000
                ctx["timings"]["score"] = elapsed
                logger.info(f"[SCORE] reranked {before_count}→{after_count} ({elapsed:.1f}ms)")
                return

            except Exception as e:
                ctx["errors"].append({"stage": "SCORE", "error": str(e), "fallback": "passthrough"})
                logger.warning(f"[SCORE] Reranker error: {e}, falling through to passthrough")

        # Passthrough fallback
        ctx["scored_candidates"] = list(candidates)
        elapsed = (time.perf_counter() - t0) * 1000
        ctx["timings"]["score"] = elapsed
        logger.info(f"[SCORE] passthrough ({elapsed:.1f}ms)")

    # ------------------------------------------------------------------
    # Stage 4: BUILD_CONTEXT
    # ------------------------------------------------------------------

    def _stage_build_context(self, ctx: Dict[str, Any]) -> None:
        t0 = time.perf_counter()
        try:
            # Mode prefix
            mode_prefix = get_mode_prefix(self.chat_service.prompts, ctx["mode"])

            # User profile
            user_profile = self.chat_service.get_user_profile_context()

            # Memories from scored candidates
            memories = ""
            if ctx["scored_candidates"]:
                memories = "\n".join(f"- {c['text']}" for c in ctx["scored_candidates"])

            # Rolling summary
            rolling_summary = ""
            if ctx["conversation_id"]:
                convo = Conversation.query.get(ctx["conversation_id"])
                if convo and convo.rolling_summary:
                    rolling_summary = convo.rolling_summary

            # Recent history
            recent_history = ""
            if ctx["recent_messages"]:
                lines = []
                for msg in ctx["recent_messages"]:
                    prefix = "User" if msg["role"] == "user" else "Assistant"
                    lines.append(f"{prefix}: {msg['content']}")
                recent_history = "\n".join(lines)

            # Current turn
            current_turn = f"User: {ctx['user_message']}\nAssistant:"

            # Web search context
            web_context = ""
            if ctx.get("web_results"):
                web_context = self.web_search.format_for_prompt(ctx["web_results"])

            # Assemble via ChatService.build_prompt
            ctx["assembled_prompt"] = self.chat_service.build_prompt(
                mode_prefix=mode_prefix,
                user_profile=user_profile,
                memories=memories,
                rolling_summary=rolling_summary,
                recent_history=recent_history,
                current_turn=current_turn,
                web_context=web_context,
            )

            ctx["context_parts"] = {
                "mode_prefix": bool(mode_prefix),
                "user_profile": bool(user_profile),
                "memories": len(ctx["scored_candidates"]),
                "rolling_summary": bool(rolling_summary),
                "recent_messages": len(ctx["recent_messages"]),
                "web_context": bool(web_context),
            }

        except Exception as e:
            ctx["assembled_prompt"] = f"User: {ctx['user_message']}\nAssistant:"
            ctx["errors"].append({"stage": "BUILD_CONTEXT", "error": str(e), "fallback": "minimal prompt"})
            logger.warning(f"[BUILD_CONTEXT] Error: {e}, using minimal prompt")

        elapsed = (time.perf_counter() - t0) * 1000
        ctx["timings"]["build_context"] = elapsed
        logger.info(f"[BUILD_CONTEXT] prompt={len(ctx['assembled_prompt'])} chars ({elapsed:.1f}ms)")

    # ------------------------------------------------------------------
    # Stage 5: GENERATE (yields SSE)
    # ------------------------------------------------------------------

    def _resolve_llm(self, model_name: Optional[str]):
        """Resolve a model display name to a BaseLLM instance."""
        if model_name and model_name in self.model_registry:
            return self.model_registry[model_name]
        if model_name:
            logger.warning(f"[GENERATE] Model '{model_name}' not in registry, falling back to local")
        return self.local_llm

    def _stage_generate(self, ctx: Dict[str, Any]) -> Generator[str, None, None]:
        t0 = time.perf_counter()
        try:
            llm = self._resolve_llm(ctx.get("model"))
            token_generator = llm.generate(
                ctx["assembled_prompt"], stream=True
            )
            for token in token_generator:
                ctx["response_text"] += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            ctx["errors"].append({"stage": "GENERATE", "error": str(e), "fallback": "error event"})
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            logger.warning(f"[GENERATE] Error: {e}")

        elapsed = (time.perf_counter() - t0) * 1000
        ctx["timings"]["generate"] = elapsed
        logger.info(f"[GENERATE] completed ({elapsed:.1f}ms)")

    # ------------------------------------------------------------------
    # Stage 6: POST_PROCESS (fire-and-forget)
    # ------------------------------------------------------------------

    def _stage_post_process(self, ctx: Dict[str, Any]) -> None:
        t0 = time.perf_counter()
        access_updated = 0
        facts_extracted = 0

        try:
            # Update access metadata for retrieved memories
            if ctx["scored_candidates"]:
                now = datetime.utcnow().isoformat()
                for candidate in ctx["scored_candidates"]:
                    try:
                        existing = self.memory_service.store.get_memory(candidate["memory_id"])
                        if existing:
                            meta = existing["metadata"]
                            new_count = meta.get("access_count", 0) + 1
                            self.memory_service.store.update_metadata(
                                candidate["memory_id"],
                                {"access_count": new_count, "last_accessed": now},
                            )
                            access_updated += 1
                    except Exception as e:
                        logger.warning(f"[POST_PROCESS] Failed to update access for {candidate['memory_id']}: {e}")

            # Extract facts from the new exchange
            if ctx["conversation_id"] and ctx["response_text"]:
                exchange_text = f"User: {ctx['user_message']}\nAssistant: {ctx['response_text']}"
                try:
                    facts_extracted = self.memory_service.process_semantic_memory(
                        ctx["conversation_id"], exchange_text, []
                    )
                except Exception as e:
                    logger.warning(f"[POST_PROCESS] Fact extraction error: {e}")

        except Exception as e:
            ctx["errors"].append({"stage": "POST_PROCESS", "error": str(e), "fallback": "skipped"})
            logger.warning(f"[POST_PROCESS] Error: {e}")

        elapsed = (time.perf_counter() - t0) * 1000
        ctx["timings"]["post_process"] = elapsed
        logger.info(
            f"[POST_PROCESS] updated {access_updated} access records, "
            f"extracted {facts_extracted} facts ({elapsed:.1f}ms)"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_pipeline_summary(self, ctx: Dict[str, Any]) -> None:
        t = ctx["timings"]
        total = sum(t.values())
        parts = " | ".join(f"{k}={v:.1f}" for k, v in t.items())
        logger.info(f"[PIPELINE] Total={total:.1f}ms | {parts}")
        if ctx["errors"]:
            logger.warning(f"[PIPELINE] Errors: {ctx['errors']}")

    def get_pipeline_metadata(self) -> Optional[Dict[str, Any]]:
        """Return diagnostics from the last pipeline run."""
        if not self._last_pipeline_ctx:
            return None
        ctx = self._last_pipeline_ctx

        # Web search diagnostics
        web_results = ctx.get("web_results")
        web_search_info = {"performed": False}
        if web_results is not None:
            web_search_info = {
                "performed": True,
                "query": web_results.get("query", ""),
                "result_count": web_results.get("result_count", 0),
                "error": web_results.get("error"),
            }

        return {
            "timings": ctx.get("timings", {}),
            "errors": ctx.get("errors", []),
            "classification": ctx.get("classification", {}),
            "candidate_count": len(ctx.get("scored_candidates", [])),
            "prompt_length": len(ctx.get("assembled_prompt", "")),
            "web_search": web_search_info,
        }
