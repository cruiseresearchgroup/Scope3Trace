#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bounded LLM interface for the Scope3Trace agents (§3.3, Appendix E).

The LLM is restricted to bounded semantic tasks (page localization, table
interpretation, entity-linking adjudication, relevance validation). All calls
run at temperature=0, on an OpenAI-compatible chat-completions endpoint so
the backbone can be swapped across model families (GPT-4o / GPT-4o-mini /
Gemini / DeepSeek — Appendix E) without touching agent code.

Per-agent wall time, token usage, and cost are metered for the Appendix D
style overhead accounting in the evaluation report.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from scope3trace.common.config import (
    DEFAULT_TEMPERATURE,
    PRICING_PER_MTOK,
    resolve_provider,
)
from scope3trace.common.state import AgentUsage


class LLMCallError(RuntimeError):
    """Raised when an experimental run forbids silent rule fallbacks."""


class LLMTruncatedError(LLMCallError):
    """The response hit the output-token ceiling; the request can be split."""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the single JSON object a bounded prompt must return."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    t = t[start:end + 1]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", t)  # trailing commas
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


class BoundedLLM:
    """Thin, deterministic (temperature=0) chat-completions client with metering."""

    def __init__(self, agent_name: str, model: str,
                 api_key: Optional[str] = None,
                 timeout: int = 180, max_retries: int = 6,
                 strict_failures: bool = False, max_json_retries: int = 2,
                 retry_delays: Optional[tuple] = None) -> None:
        self.agent_name = agent_name
        self.model = model
        provider = resolve_provider(model)
        self.base_url = provider["base_url"]
        self.api_key = api_key or os.environ.get(provider["api_key_env"]) or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.strict_failures = strict_failures
        self.max_json_retries = max_json_retries
        self.retry_delays = retry_delays
        self.usage = AgentUsage(agent=agent_name, model=model)
        self._disabled_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) and self._disabled_reason is None

    @property
    def disabled_reason(self) -> Optional[str]:
        if self._disabled_reason:
            return self._disabled_reason
        if not self.api_key:
            return f"API key is not configured for {self.model}"
        return None

    def _fail(self, reason: str) -> None:
        self._disabled_reason = reason
        if self.strict_failures:
            raise LLMCallError(f"{self.agent_name}/{self.model}: {reason}")
        print(f"[WARN] {self.agent_name} LLM disabled: {reason} — continuing with rule fallbacks")

    def _price(self, tokens_in: int, tokens_out: int) -> float:
        p = PRICING_PER_MTOK.get(self.model.lower())
        if not p:
            for key, val in PRICING_PER_MTOK.items():
                if self.model.lower().startswith(key):
                    p = val
                    break
        if not p:
            return 0.0
        return tokens_in / 1e6 * p["in"] + tokens_out / 1e6 * p["out"]

    def complete_json(self, system: str, user: str,
                      max_tokens: int = 2000, images: Optional[list] = None,
                      extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """One bounded call; returns the parsed JSON object (or None on failure).

        ``images``: optional base64 PNG page renderings for a vision-capable
        backbone (chart reading); sent as OpenAI-compatible image_url parts.
        """
        if not self.available:
            if self.strict_failures:
                raise LLMCallError(
                    f"{self.agent_name}/{self.model}: {self.disabled_reason or 'unavailable'}")
            return None
        try:
            import requests
        except ImportError:
            self._fail("requests package is not installed")
            return None
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user if not images else
                 [{"type": "text", "text": user}] +
                 [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in images]},
            ],
            "response_format": {"type": "json_object"},
        }
        if extra:
            payload.update(extra)  # provider options, e.g. reasoning_effort
        t0 = time.time()
        last_err: Optional[str] = None
        json_failures = 0
        self.last_truncated = False
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = f"request error: {e}"
                if attempt < self.max_retries:
                    delay = (self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                             if self.retry_delays else min(2 ** attempt, 30))
                    time.sleep(delay)
                continue
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage") or {}
                self.usage.calls += 1
                self.usage.tokens_in += int(usage.get("prompt_tokens") or 0)
                self.usage.tokens_out += int(usage.get("completion_tokens") or 0)
                self.usage.cost_usd = round(self._price(self.usage.tokens_in, self.usage.tokens_out), 6)
                self.usage.wall_time_s += time.time() - t0
                t0 = time.time()  # do not count earlier attempts twice
                try:
                    content = data["choices"][0]["message"]["content"]
                    parsed = extract_json_object(content) if isinstance(content, str) else None
                except (KeyError, IndexError, TypeError):
                    parsed = None
                truncated = (data.get("choices") or [{}])[0].get("finish_reason") == "length"
                if truncated:
                    parsed = None  # even a closed JSON prefix is not a complete extraction
                self.usage.response_metadata.append({
                    "model": data.get("model"),
                    "system_fingerprint": data.get("system_fingerprint"),
                    "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
                    "max_tokens": max_tokens,
                    "attempt": attempt,
                    "valid_json_object": parsed is not None,
                })
                audit_dir = os.environ.get("SCOPE3TRACE_LLM_AUDIT_DIR")
                if audit_dir:
                    target = Path(audit_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    # Never persist authorization headers or API credentials.
                    logged = payload
                    if images:
                        import copy, hashlib as _h
                        logged = copy.deepcopy(payload)
                        for part in logged["messages"][1]["content"]:
                            if part.get("type") == "image_url":
                                u = part["image_url"]["url"]
                                part["image_url"]["url"] = "sha256:" + _h.sha256(u.encode()).hexdigest()
                    (target / f"{self.agent_name}-{self.usage.calls:04d}.json").write_text(
                        json.dumps({"request": logged, "response": data,
                                    "attempt": attempt, "valid_json_object": parsed is not None},
                                   ensure_ascii=False),
                        encoding="utf-8")
                if parsed is None:
                    if truncated:
                        # Repeating the same capped request cannot recover its
                        # missing tail. Preserve the response and fail explicitly.
                        last_err = f"response truncated at output limit {max_tokens}"
                        self.last_truncated = True
                        break
                    json_failures += 1
                    last_err = "response was not a valid JSON object"
                    # Format-only retry of the identical request, bounded by
                    # both limits. Valid negative semantic verdicts never retry.
                    if json_failures <= self.max_json_retries:
                        continue
                    break
                return parsed
            if resp.status_code in (429, 500, 502, 503, 504):
                body = resp.text.replace(self.api_key, "[REDACTED]")[:500]
                if any(tag in body for tag in
                       ("insufficient_quota", "credit_balance_exhausted", "billing", "PerDay", "per day",
                        "RequestsPerDay", "daily")):
                    # permanent condition — retrying would stall the pipeline;
                    # disable this backbone so agents fall back to rule paths
                    self.usage.wall_time_s += time.time() - t0
                    self._fail(f"quota exhausted for {self.model}")
                    return None
                retry_after = resp.headers.get("retry-after")
                wait = (self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                        if self.retry_delays else min(2 ** attempt, 60))
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                if attempt < self.max_retries:
                    time.sleep(wait)
                last_err = f"HTTP {resp.status_code}: {body}"
                continue
            if resp.status_code == 400 and "response_format" in resp.text:
                payload.pop("response_format", None)  # provider without JSON mode
                continue
            if resp.status_code == 400 and extra and any(k in resp.text for k in extra):
                for k in extra:
                    payload.pop(k, None)  # provider without this option
                continue
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            break
        self.usage.wall_time_s += time.time() - t0
        if self.strict_failures:
            cls = LLMTruncatedError if self.last_truncated else LLMCallError
            raise cls(
                f"{self.agent_name}/{self.model}: LLM call failed: {last_err}")
        print(f"[WARN] {self.agent_name} LLM call failed: {last_err}")
        return None
