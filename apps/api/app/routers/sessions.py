"""Sessions, messages, and the chat stream.

Ordering rules for a streaming turn:

* user message committed before the upstream call
* assistant message and credit deduction committed together
* no charge for a turn that produced no output
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, or_
from sqlmodel import col, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.deps import CurrentUser, DbSession, client_ip
from app.models.chat import (
    ChatSession,
    Message,
    Role,
    RoutingMode,
    SessionKind,
    TurnFailure,
)
from app.models.user import AuditEvent, User, utcnow
from app.models.workspace import Agent as WorkspaceAgent
from app.models.workspace import (
    AgentVisibility,
    Artifact,
    ArtifactKind,
    ArtifactVersion,
    Job,
    Memory,
    MemoryType,
    Project,
    Skill,
    SkillSource,
    StoredFile,
)
from app.models.workspace import Visibility as SkillVisibility
from app.schemas.auth import Preferences
from app.schemas.chat import (
    AudioRequest,
    ChooseVariant,
    CompareRequest,
    DiagramOut,
    DiagramRequest,
    DiagramStore,
    FigureSuggestion,
    FigureSuggestRequest,
    ImageRequest,
    MessageOut,
    SendMessage,
    SessionBulkDelete,
    SessionCreate,
    SessionMade,
    SessionOut,
    SessionPatch,
    made_from_artifacts,
    snippet,
)
from app.schemas.workspace import ArtifactOut
from app.services import (
    adaptive_routing,
    artifact_extract,
    audiogen,
    chart_code,
    design_templates,
    figures,
    governance,
    grounding,
    imagegen,
    index_client,
    lint,
    revise,
    richtext,
    settings_store,
)
from app.services import agent as agent_service
from app.services import auto_memory as auto_memory_service
from app.services import chat as chat_service
from app.services import deck as deck_service
from app.services import design as design_service
from app.services import (
    diagram as diagram_service,
)
from app.services import files as file_service
from app.services import litellm as litellm_service
from app.services import models as model_service
from app.services import page as page_service
from app.services import report as report_service
from app.services.context import build_messages, requests_web_search, with_pictures
from app.services.credits import charge_for_tokens, has_headroom, settle
from app.services.tools.base import Tool, ToolContext, openai_snapshot
from app.services.tools.registry import build_tools
from app.services.workspace_context import (
    ContextBlock,
    ContextFile,
    WorkspaceContext,
    WorkspaceContextError,
    agent_settings,
    assemble,
    design_for,
    file_budget,
    reads_pictures,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


@dataclass(slots=True)
class _PrivacyResolution:
    models: list[dict]
    action: str
    findings: list[governance.Finding]
    routing: dict[str, Any]
    mask_outbound: bool = False

    @property
    def strict_local(self) -> bool:
        return bool(self.models) and all(
            model.get("dataBoundary") == "self_hosted" and model.get("strictLocal") is True
            for model in self.models
        )


def _strict_model(model: dict) -> bool:
    return model.get("dataBoundary") == "self_hosted" and model.get("strictLocal") is True


_STRICT_LOCAL_TOOL_NAMES = frozenset(
    {
        # No network egress; a "builtin" source alone is not proof of that.
        "search_knowledge",
        "create_artifact",
        "create_chart",
    }
)


async def _knowledge_shelf(
    db: AsyncSession, user: User, session: ChatSession, agent: WorkspaceAgent | None
) -> tuple[list[tuple[str, str, str | None]], str]:
    """Documents `search_knowledge` can look through this turn, and their index collection.

    An agent's knowledge and the files uploaded into this conversation, whole text
    because the tool runs inside the stream with no DB session. The collection is
    the agent's when there is one — the tool searches one — else the conversation's.
    """
    rows = (
        await db.exec(
            select(StoredFile)
            .where(
                StoredFile.user_id == user.id,
                or_(
                    StoredFile.session_id == session.id,
                    StoredFile.agent_id == (session.agent_id or ""),
                ),
            )
            .order_by(col(StoredFile.created_at))
        )
    ).all()
    shelf = [
        (row.name, row.text, row.source_url)
        for row in rows
        if row.text
        and (
            (row.agent_id and row.agent_id == session.agent_id)
            or (row.session_id == session.id and row.project_id is None and row.agent_id is None)
        )
    ]
    if session.agent_id:
        return shelf, (agent.index_key or "") if agent else ""
    return shelf, session.index_key or ""


def _strict_local_tools(tools: list[Tool]) -> list[Tool]:
    """Keeps only tools whose implementation has no network egress.

    An admin-configured HTTP endpoint is not proof its arguments stay inside the boundary.
    """
    return [
        tool for tool in tools if tool.source == "builtin" and tool.name in _STRICT_LOCAL_TOOL_NAMES
    ]


def _serialized_tool_definitions(definitions: list[dict[str, Any]]) -> str:
    """Stable text for both privacy inspection and decision-token hashing."""
    return json.dumps(
        definitions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _tool_definition_source(tools: list[Tool]) -> str:
    """The exact ordered schema array the OpenAI request would carry."""
    return _serialized_tool_definitions(openai_snapshot(tools))


def _drop_sensitive_tool_definitions(tools: list[Tool], *, legacy: bool = False) -> list[Tool]:
    """Drops whole tools whose schema carries a finding.

    Renaming a schema key would desync the runner registry.
    """
    return [
        tool
        for tool in tools
        if not governance.findings(
            {"tool_definitions": _tool_definition_source([tool])},
            legacy=legacy,
        )
    ]


def _combined_boundary(models: list[dict]) -> str:
    boundaries = {str(model.get("dataBoundary") or "unknown") for model in models}
    return next(iter(boundaries)) if len(boundaries) == 1 else "mixed"


def _with_actual_model(routing: dict, routed_model: str, actual_model: str) -> dict:
    """Returns routing metadata updated from LiteLLM's provider-reported id."""
    actual = list(routing.get("actualModels") or [])
    if actual_model not in actual:
        actual.append(actual_model)
    routes = []
    matched = False
    for route in routing.get("modelRoutes") or []:
        if route.get("routedModel") == routed_model:
            routes.append({**route, "actualModel": actual_model})
            matched = True
        else:
            routes.append(route)
    if not matched:
        routes.append(
            {
                "routedModel": routed_model,
                "actualModel": actual_model,
                "dataBoundary": "unknown",
            }
        )
    updated = {
        **routing,
        "actualModels": actual,
        "modelRoutes": routes,
    }
    if len(routing.get("routedModels") or []) == 1:
        updated["actualModel"] = actual_model
    else:
        updated.pop("actualModel", None)
    return updated


def _allowed_models(user: User, catalogue: list[dict], *, kind: str) -> list[dict]:
    """Models allowed for this account and surface; an empty allowlist means the whole catalogue."""
    allowed = set(user.allowed_models or [])
    return [
        model
        for model in catalogue
        if kind in model.get("kinds", []) and (not allowed or model.get("id") in allowed)
    ]


#: Boundary rule for the outline model; the rule lives in `adaptive_routing`.
_widens_boundary = adaptive_routing.widens_boundary

#: Both Auto lanes; every Auto gate applies to both.
_AUTO_MODES = frozenset({RoutingMode.auto, RoutingMode.auto_quality})


def _planner_model(
    wanted: str | None,
    *,
    user: User,
    catalogue: list[dict],
    kind: str,
    writer: dict,
    strict_local: bool,
) -> dict | None:
    """Catalogue row for the outline call, or `None` to use the writer.

    Refused when the allowlist, surface, strict-local route or writer boundary would not allow it.
    """
    if not wanted or strict_local:
        return None
    planner = model_service.find(_allowed_models(user, catalogue, kind=kind), str(wanted))
    if planner is None or _widens_boundary(planner, writer):
        log.info("outline model %s unusable here", wanted)
        return None
    return planner


async def _enrichment_model(writer: dict, *, strict_local: bool, disable_fallbacks: bool) -> dict:
    """Catalogue row that titles the session and extracts memories.

    The writer itself when `title_model` is unset.
    """
    if strict_local or disable_fallbacks:
        return writer
    resolved = await model_service.resolve_enrichment_model()
    if not resolved or resolved == writer["id"]:
        return writer
    catalogue = await model_service.list_models()
    return model_service.find(catalogue["models"], resolved) or writer


def _find_auto_quality_model(models: list[dict], model_id: str | None) -> dict | None:
    """Finds an answer model; privacy-only capacity is classifier-only."""
    model = model_service.find(models, model_id or "")
    return model if model is not None and model.get("privacyOnly") is not True else None


async def _require_auto_quality_model(user: User, model_id: str | None) -> None:
    """Requires Auto's quality ceiling to be live, allowed and chat-capable."""
    catalogue = await model_service.list_models_for_egress()
    usable = _allowed_models(user, catalogue.get("models", []), kind=SessionKind.chat.value)
    if _find_auto_quality_model(usable, model_id) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="auto_quality_model_required",
        )


def _cost_routing(
    *,
    decision: str,
    reason_code: str,
    requested_model: dict,
    selected_model: dict,
    mode: str = "auto",
    classifier_model: str | None = None,
    classification: adaptive_routing.Classification | None = None,
) -> dict[str, Any]:
    """One turn's routing decision.

    Wire key is `costRouting` for both lanes; `mode` names the lane.
    """
    route: dict[str, Any] = {
        "mode": mode,
        "decision": decision,
        "reasonCode": reason_code,
        "requestedModel": requested_model["id"],
        "selectedModel": selected_model["id"],
        "classifierVersion": adaptive_routing.CLASSIFIER_VERSION,
    }
    if classifier_model:
        route["classifierModel"] = classifier_model
    if classification is not None:
        route.update(
            {
                "complexity": classification.complexity,
                "confidence": classification.confidence,
                "classifierInputTokens": classification.input_tokens,
                "classifierOutputTokens": classification.output_tokens,
            }
        )
    return route


async def _resolve_cost_routing(
    *,
    #: Stored as a plain string, so compared by value.
    mode: RoutingMode | str = RoutingMode.auto,
    db: DbSession,
    user: User,
    policy,
    catalogue: list[dict],
    quality_model: dict,
    classifier_messages: list[dict[str, str]],
    classifier_tool_definitions: list[dict[str, Any]],
    context_tokens: int,
    unsupported_reason: str | None,
) -> tuple[dict, dict[str, Any]]:
    """An Auto turn's effective model and value-free route metadata.

    Cost acts on `low`, quality on `high`; any refusal keeps the chosen model.
    """
    lane = str(getattr(mode, "value", mode))
    upgrading = lane == RoutingMode.auto_quality.value
    # Each lane has its own switch.
    lane_enabled = policy.adaptive_quality_enabled if upgrading else policy.adaptive_routing_enabled
    if not lane_enabled:
        return quality_model, _cost_routing(
            mode=lane,
            decision="bypassed",
            reason_code="disabled",
            requested_model=quality_model,
            selected_model=quality_model,
        )
    if unsupported_reason:
        return quality_model, _cost_routing(
            mode=lane,
            decision="bypassed",
            reason_code=unsupported_reason,
            requested_model=quality_model,
            selected_model=quality_model,
        )

    allowed = set(user.allowed_models or [])
    classifier_id = str(policy.adaptive_classifier_model_id or "")
    classifier_model = model_service.find(catalogue, classifier_id)
    if not adaptive_routing.classifier_is_usable(classifier_model, allowed_model_ids=allowed):
        return quality_model, _cost_routing(
            mode=lane,
            decision="classifier_unavailable",
            reason_code="classifier_unavailable",
            requested_model=quality_model,
            selected_model=quality_model,
            classifier_model=classifier_id or None,
        )

    if upgrading:
        candidates = adaptive_routing.quality_candidates(
            catalogue,
            list(policy.adaptive_quality_model_ids or [])[:3],
            quality_model=quality_model,
            allowed_model_ids=allowed,
            context_tokens=context_tokens,
            # Upgraded turns keep their tools.
            requires_tools=bool(classifier_tool_definitions),
        )
    else:
        candidates = adaptive_routing.economy_candidates(
            catalogue,
            list(policy.adaptive_economy_model_ids or [])[:3],
            quality_model=quality_model,
            allowed_model_ids=allowed,
            context_tokens=context_tokens,
            # Economy turns run tool-free, so the preflight context fit holds.
            requires_tools=False,
        )
    if not candidates:
        return quality_model, _cost_routing(
            mode=lane,
            decision="kept_quality",
            reason_code="no_quality_model" if upgrading else "no_economy_model",
            requested_model=quality_model,
            selected_model=quality_model,
            classifier_model=classifier_id,
        )

    classifier_input = adaptive_routing.classifier_context(
        classifier_messages,
        classifier_tool_definitions,
    )
    if classifier_input is None:
        return quality_model, _cost_routing(
            mode=lane,
            decision="kept_quality",
            reason_code="input_too_long",
            requested_model=quality_model,
            selected_model=quality_model,
            classifier_model=classifier_id,
        )

    # Not `credentials_for`: it may fall back to the master key. The classifier
    # needs the user's own key.
    api_key = litellm_service.user_key(user) or await litellm_service.ensure_key(user)
    if not api_key:
        return quality_model, _cost_routing(
            mode=lane,
            decision="classifier_unavailable",
            reason_code="classifier_key_unavailable",
            requested_model=quality_model,
            selected_model=quality_model,
            classifier_model=classifier_id,
        )
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    classification = await adaptive_routing.classify(
        model_id=classifier_id,
        context=classifier_input,
        user_id=user.id,
        api_key=api_key,
    )
    if classification is None:
        return quality_model, _cost_routing(
            mode=lane,
            decision="classifier_unavailable",
            reason_code="classifier_unavailable",
            requested_model=quality_model,
            selected_model=quality_model,
            classifier_model=classifier_id,
        )
    wanted = "high" if upgrading else "low"
    bar = adaptive_routing.MIN_HIGH_CONFIDENCE if upgrading else adaptive_routing.MIN_LOW_CONFIDENCE
    if classification.complexity != wanted or classification.confidence < bar:
        if classification.complexity == "uncertain":
            reason = "uncertain"
        elif classification.complexity != wanted:
            reason = f"{classification.complexity}_complexity"
        else:
            reason = "low_confidence"
        return quality_model, _cost_routing(
            mode=lane,
            decision="kept_quality",
            reason_code=reason,
            requested_model=quality_model,
            selected_model=quality_model,
            classifier_model=classifier_id,
            classification=classification,
        )

    selected = candidates[0]
    return selected, _cost_routing(
        mode=lane,
        decision="routed",
        reason_code=f"{wanted}_complexity",
        requested_model=quality_model,
        selected_model=selected,
        classifier_model=classifier_id,
        classification=classification,
    )


def _apply_effective_model(routing: dict[str, Any], model: dict) -> dict[str, Any]:
    """Updates privacy routing after a clean Auto decision."""
    model_id = model["id"]
    return {
        **routing,
        "routedModels": [model_id],
        "effectiveModels": [model_id],
        "actualModels": [],
        "dataBoundary": model.get("dataBoundary") or "unknown",
        "modelRoutes": [
            {
                "routedModel": model_id,
                "actualModel": None,
                "dataBoundary": model.get("dataBoundary") or "unknown",
            }
        ],
    }


def _substitution_routing(requested: dict, effective: dict) -> dict[str, Any]:
    """Routing metadata for a fallback made outside the privacy decision (report and slides)."""
    effective_id = effective["id"]
    boundary = effective.get("dataBoundary") or "unknown"
    return {
        "requestedModels": [requested["id"]],
        "routedModels": [effective_id],
        "effectiveModels": [effective_id],
        # Document runners never see a provider-reported id.
        "actualModels": [effective_id],
        "actualModel": effective_id,
        "action": "none",
        "dataBoundary": boundary,
        "modelRoutes": [
            {
                "routedModel": effective_id,
                "actualModel": effective_id,
                "dataBoundary": boundary,
            }
        ],
    }


def _privacy_sources(
    content: str,
    history: list[Message],
    blocks: tuple[ContextBlock, ...] | list[ContextBlock] | None = None,
) -> dict[str, str | list[str]]:
    sources: dict[str, str | list[str]] = {
        "current_input": content,
        "conversation_history": [message.content for message in history],
    }
    source_kinds = {
        "user.instructions": "user_instructions",
        "agent.instructions": "agent",
        "project.instructions": "project_instructions",
        "project.design": "project_design",
        "attachment": "attachments",
        "project.knowledge": "project_knowledge",
        "memory": "memory",
    }
    for block in blocks or []:
        source = (
            "skills"
            if block.source.startswith("skill:")
            else source_kinds.get(block.source, block.source)
        )
        existing = sources.get(source)
        if existing is None:
            sources[source] = [block.text]
        elif isinstance(existing, list):
            existing.append(block.text)
    return sources


def _privacy_response(
    *,
    rows: list[governance.Finding],
    requested: list[dict],
    safe: list[dict],
    token: str,
    allow_raw: bool,
) -> JSONResponse:
    actions = ["mask_external"]
    if safe:
        actions.insert(0, "route_strict_local")
    if allow_raw:
        actions.append("send_raw_external")
    actions.append("edit")
    actions.append("cancel")
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "code": "privacy_decision_required",
            "findings": [row.wire() for row in rows],
            "requestedModels": [model["id"] for model in requested],
            "safeModels": [{"id": model["id"], "label": model["label"]} for model in safe],
            "allowedActions": actions,
            "decisionToken": token,
            "detectorVersion": governance.DETECTOR_VERSION,
            "policyVersion": governance.POLICY_VERSION,
        },
    )


async def _resolve_privacy(
    *,
    user: User,
    session: ChatSession,
    policy,
    catalogue: list[dict],
    requested: list[dict],
    sources: dict[str, str | list[str]],
    explicit_action: str | None,
    decision_token: str | None,
) -> _PrivacyResolution | JSONResponse:
    """Makes one egress decision before persistence, billing or upstream.

    Only explicit proxy metadata makes a safe candidate; unknown boundaries count as external.
    """
    rows = (
        governance.findings(sources, legacy=policy.pii_masking)
        if (policy.external_data_guard or policy.pii_masking)
        else []
    )
    requested_ids = [model["id"] for model in requested]
    digest = governance.envelope_digest(sources)

    allowed_models = set(user.allowed_models or [])
    safe: list[dict] = []
    safe_ids = set(policy.privacy_safe_model_ids or [])
    # Priority is catalogue order, the order the admin screen shows.
    for model in catalogue:
        model_id = model.get("id")
        if (
            model_id in safe_ids
            and _strict_model(model)
            and "chat" in model.get("kinds", [])
            and (not allowed_models or model_id in allowed_models)
        ):
            safe.append(model)

    action = "none"
    effective = requested
    mask_outbound = False
    all_strict = all(_strict_model(model) for model in requested)
    allow_raw = policy.allow_user_raw_external and not policy.pii_masking

    if rows and policy.pii_masking:
        # The legacy organisation-wide setting is always the strongest rule.
        action = "mask_external"
        mask_outbound = True
    elif rows and all_strict:
        action = "strict_local"
    elif rows and policy.external_data_guard:
        preference = Preferences.of(user).privacy_default_action
        chosen = explicit_action or preference
        if chosen == "send_raw_external" and not allow_raw:
            chosen = "ask"
        if chosen == "route_strict_local" and not safe:
            chosen = "ask"

        # A request-supplied choice is a retry and must match the exact user,
        # session, models and envelope that produced the warning.
        if explicit_action and not governance.verify_decision_token(
            decision_token,
            user_id=user.id,
            session_id=session.id,
            requested_models=requested_ids,
            digest=digest,
        ):
            chosen = "ask"

        if chosen == "route_strict_local":
            action = chosen
            # A comparison collapses to one strict-local call.
            effective = [safe[0]]
        elif chosen == "mask_external":
            action = chosen
            mask_outbound = True
        elif chosen == "send_raw_external" and allow_raw:
            action = chosen
        else:
            token = governance.issue_decision_token(
                user_id=user.id,
                session_id=session.id,
                requested_models=requested_ids,
                digest=digest,
            )
            return _privacy_response(
                rows=rows,
                requested=requested,
                safe=safe,
                token=token,
                allow_raw=allow_raw,
            )

    routing = {
        "requestedModels": requested_ids,
        "routedModels": [model["id"] for model in effective],
        "effectiveModels": [model["id"] for model in effective],
        "actualModels": [],
        "action": action,
        "dataBoundary": _combined_boundary(effective),
        "modelRoutes": [
            {
                "routedModel": model["id"],
                "actualModel": None,
                "dataBoundary": model.get("dataBoundary") or "unknown",
            }
            for model in effective
        ],
        "detectorVersion": governance.DETECTOR_VERSION,
        "policyVersion": governance.POLICY_VERSION,
        "findingCounts": [row.wire() for row in rows],
        "compareCollapsed": len(requested) > 1 and len(effective) == 1,
    }
    return _PrivacyResolution(effective, action, rows, routing, mask_outbound)


def _decision_audit_metadata(response: JSONResponse, session_id: str) -> dict[str, Any]:
    """Extracts only the value-free fields from a privacy 409 contract."""
    contract = json.loads(response.body)
    return {
        "sessionId": session_id,
        "policyVersion": contract["policyVersion"],
        "detectorVersion": contract["detectorVersion"],
        "findings": contract["findings"],
        "requestedModels": contract["requestedModels"],
        "action": "decision_required",
    }


def _mask_list(values: list[str], *, legacy: bool = False) -> list[str]:
    masker = governance.mask_legacy if legacy else governance.mask
    return [masker(value)[0] for value in values]


def _mask_text_tree(value: Any, masker) -> Any:
    """Returns a detached tree with every persisted string sanitized."""
    if isinstance(value, str):
        return masker(value)[0]
    if isinstance(value, dict):
        return {key: _mask_text_tree(item, masker) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_text_tree(item, masker) for item in value]
    if isinstance(value, tuple):
        return tuple(_mask_text_tree(item, masker) for item in value)
    return value


async def _require_egress_policy():
    """Loads the authoritative policy or refuses before any other egress work."""
    try:
        return await governance.current_for_egress()
    except governance.GovernanceUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="governance_unavailable",
        ) from None


async def _owned(db: DbSession, user: User, session_id: str) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    # Same answer for "missing" and "someone else's".
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    return session


async def _history(db: DbSession, session_id: str) -> list[Message]:
    result = await db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(col(Message.created_at), col(Message.id))
    )
    return list(result.all())


def _raise_workspace_error(exc: WorkspaceContextError) -> None:
    code = str(exc)
    missing = code in {
        "agent_not_found",
        "project_not_found",
        "attachment_not_found",
        "starting_template_not_found",
    }
    raise HTTPException(
        status_code=(
            status.HTTP_404_NOT_FOUND
            if missing
            # Literal 422: the Starlette constant name differs across supported versions.
            else 422
        ),
        detail=code,
    ) from exc


async def _validate_session_links(
    db: DbSession,
    user: User,
    kind: SessionKind,
    *,
    project_id: str | None,
    agent_id: str | None,
) -> None:
    if project_id:
        project = await db.get(Project, project_id)
        if project is None or project.user_id != user.id:
            raise HTTPException(status_code=404, detail="project_not_found")
    if agent_id:
        agent = await db.get(WorkspaceAgent, agent_id)
        allowed = agent is not None and (
            agent.owner_id == user.id or agent.visibility == AgentVisibility.org
        )
        if not allowed:
            raise HTTPException(status_code=404, detail="agent_not_found")
        if not agent.enabled:
            raise HTTPException(status_code=422, detail="agent_disabled")
        if agent.kinds and kind.value not in agent.kinds:
            raise HTTPException(status_code=422, detail="agent_kind_mismatch")


def _skill_step(event: dict | None) -> dict | None:
    if not event:
        return None
    names = [str(skill.get("name") or "") for skill in event.get("skills") or []]
    return {
        "id": "skills-applied",
        "type": "thinking",
        "label": f"스킬 {len(names)}개 적용",
        "status": "done",
        # Structured fields beside the display strings, for audit.
        "skills": list(event.get("skills") or []),
        "estimatedTokens": int(event.get("estimatedTokens") or 0),
        "detail": (" · ".join(names) + f" · 약 {int(event.get('estimatedTokens') or 0):,} 토큰"),
    }


#: Per-file detail line, keyed by `ContextFile.state`.
_FILE_NOTE = {
    "truncated": "{name} {kept:,}자만 반영",
    "omitted": "{name} 분량을 넘겨 제외",
    "unreadable": "{name} 읽지 못함",
    # Readable file, but this model cannot see pictures.
    "picture_unseen": "{name} 그림 · 이 모델은 보지 못함",
}

#: Names listed before the rest are counted.
_NAMES_SHOWN = 6


def _named(notes: list[str], unit: str) -> str:
    """A detail line that names what it can and counts the rest."""
    shown = notes[:_NAMES_SHOWN]
    line = " · ".join(shown)
    if len(notes) > len(shown):
        line += f" 외 {len(notes) - len(shown)}{unit}"
    return line


def _file_context_step(step_id: str, subject: str, files: tuple[ContextFile, ...]) -> dict | None:
    if not files:
        return None
    # A picture the model saw counts as included.
    whole = {"included", "picture"}
    short = [file for file in files if file.state not in whole]
    cut = sum(1 for file in short if file.state == "truncated")
    dropped = len(short) - cut
    fates = [f"{cut}개 잘림"] if cut else []
    if dropped:
        fates.append(f"{dropped}개 빠짐")
    label = (
        f"{subject} {len(files)}개 중 " + ", ".join(fates)
        if fates
        else f"{subject} {len(files)}개 반영"
    )
    notes = [_FILE_NOTE[file.state].format(name=file.name, kept=file.kept_chars) for file in short]
    detail = _named(notes or [file.name for file in files], "개")
    return {
        "id": step_id,
        "type": "thinking",
        "label": label,
        "status": "done",
        "detail": detail,
        # Structured beside the display strings, for audit.
        "files": [
            {
                "name": file.name,
                "state": file.state,
                "keptChars": file.kept_chars,
                "totalChars": file.total_chars,
            }
            for file in files
        ],
    }


def _memory_context_step(workspace: WorkspaceContext) -> dict | None:
    names = list(workspace.loaded_memories)
    if not names:
        return None
    detail = _named(names, "건")
    if workspace.total_memories > len(names):
        detail += f" · 저장된 {workspace.total_memories}건 중 최근 {len(names)}건"
    return {
        "id": "context-memories",
        "type": "thinking",
        "label": f"메모리 {len(names)}건 참고",
        "status": "done",
        "detail": detail,
        # Names only, never bodies.
        "memories": names,
        # The client re-renders this line in the reader's language.
        "totalMemories": workspace.total_memories,
    }


def _personal_context_step(workspace: WorkspaceContext) -> dict | None:
    """One line saying personal settings shaped the turn: which half, never the text."""
    block = next((b for b in workspace.blocks if b.source == "user.instructions"), None)
    if block is None:
        return None
    parts = []
    if "# 사용자에 대해" in block.text:
        parts.append("나에 대해")
    if "# 사용자가 바라는 답변 방식" in block.text:
        parts.append("답변 방식")
    return {
        "id": "context-personal",
        "type": "thinking",
        "label": "개인 맞춤 설정 적용",
        "status": "done",
        "detail": " · ".join(parts),
        "personal": parts,
    }


def _context_steps(workspace: WorkspaceContext) -> list[dict]:
    """Timeline lines for context handed to the model silently: memories, attachments, knowledge."""
    steps = [
        _personal_context_step(workspace),
        _memory_context_step(workspace),
        _file_context_step("context-attachments", "첨부", workspace.attachments),
        _file_context_step("context-earlier", "이전 첨부", workspace.carried),
        _file_context_step("context-knowledge", "프로젝트 지식", workspace.knowledge),
    ]
    return [step for step in steps if step]


def _memory_saved_step(written: int) -> dict:
    return {
        "id": "memory-saved",
        "type": "thinking",
        "label": f"메모리 {written}건 저장",
        "status": "done",
        "detail": "자동 메모리에 추가됨",
        "memoriesWritten": written,
    }


def _step_event(step: dict) -> dict:
    """A stored step addressed for the wire.

    `type` becomes the event name; the display category rides as `category`.
    """
    return {**step, "type": "step", "category": step["type"]}


def _prelude_steps(skills_event: dict | None, context_steps: list[dict] | None) -> list[dict]:
    """Steps a turn opens with: applied skills, then context."""
    applied = _skill_step(skills_event)
    return ([applied] if applied else []) + list(context_steps or [])


async def _owned_attachments(
    db: DbSession, user: User, attachment_ids: list[str] | None
) -> tuple[list[StoredFile], list[dict] | None]:
    """Resolves every requested upload without silently dropping foreign ids."""
    if not attachment_ids:
        return [], None
    ids = list(dict.fromkeys(attachment_ids))
    found = (
        await db.exec(
            select(StoredFile).where(
                col(StoredFile.id).in_(ids),
                StoredFile.user_id == user.id,
            )
        )
    ).all()
    by_id = {row.id: row for row in found}
    if len(by_id) != len(ids):
        # Same response for a missing id and another user's id.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="attachment_not_found",
        )
    rows = [by_id[file_id] for file_id in ids]
    metadata = [
        {
            "id": row.id,
            "name": row.name,
            "size": row.size,
            "type": row.mime,
            "error": row.error,
        }
        for row in rows
    ]
    return rows, metadata


def _worth_listing(rows: list[ChatSession], empty: set[str]) -> list[ChatSession]:
    """List non-empty sessions and recently created in-flight sessions."""
    fresh = utcnow() - timedelta(minutes=1)
    return [
        row
        for row in rows
        if row.id not in empty or row.artifact_id or row.pinned or row.created_at > fresh
    ]


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    user: CurrentUser,
    db: DbSession,
    kind: SessionKind | None = None,
    project_id: str | None = None,
):
    query = select(ChatSession).where(ChatSession.user_id == user.id)
    if kind is not None:
        query = query.where(ChatSession.kind == kind)
    if project_id is not None:
        query = query.where(ChatSession.project_id == project_id)
    # Sidebar order: pinned first, then most recently touched.
    query = query.order_by(col(ChatSession.pinned).desc(), col(ChatSession.updated_at).desc())
    rows = (await db.exec(query)).all()

    ids = [s.id for s in rows]
    previews = await _previews(db, ids)
    # Absent from `_previews` means no messages.
    rows = _worth_listing(rows, {sid for sid in ids if sid not in previews})
    ids = [s.id for s in rows]
    # A media answer has an artifact and an empty body.
    made = await _made(db, [sid for sid in ids if not previews.get(sid, (None, 0))[0]])
    return [
        SessionOut.of(
            s,
            preview=previews.get(s.id, (None, 0))[0],
            message_count=previews.get(s.id, (None, 0))[1],
            made=made.get(s.id),
        )
        for s in rows
    ]


async def _previews(db: DbSession, session_ids: list[str]) -> dict[str, tuple[str | None, int]]:
    """`{session_id: (latest message snippet, message count)}`.

    Absent for a session with no messages.
    """
    if not session_ids:
        return {}
    counts = dict(
        (
            await db.exec(
                select(Message.session_id, func.count())
                .where(col(Message.session_id).in_(session_ids))
                .group_by(col(Message.session_id))
            )
        ).all()
    )
    # DISTINCT ON: newest row per conversation, no per-row subquery. Postgres-only.
    latest = (
        await db.exec(
            select(Message.session_id, Message.content)
            .where(col(Message.session_id).in_(session_ids))
            .order_by(col(Message.session_id), col(Message.created_at).desc())
            .distinct(col(Message.session_id))
        )
    ).all()
    return {sid: (snippet(content), counts.get(sid, 0)) for sid, content in latest}


async def _made(db: DbSession, session_ids: list[str]) -> dict[str, SessionMade]:
    """`{session_id: what it produced}` for sessions with no message text (pictures, clips)."""
    if not session_ids:
        return {}
    rows = (
        await db.exec(
            select(Artifact.session_id, Artifact.kind, Artifact.data)
            .where(col(Artifact.session_id).in_(session_ids))
            .order_by(col(Artifact.session_id), col(Artifact.created_at).desc())
        )
    ).all()
    by_session: dict[str, list[tuple[str, dict | None]]] = {}
    for session_id, kind, data in rows:
        if session_id is not None:
            by_session.setdefault(session_id, []).append((str(kind), data))
    summarised = {sid: made_from_artifacts(made) for sid, made in by_session.items()}
    return {sid: made for sid, made in summarised.items() if made is not None}


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: str, user: CurrentUser, db: DbSession):
    session = await _owned(db, user, session_id)
    history = await _history(db, session_id)
    made = {} if any(m.content for m in history) else await _made(db, [session_id])
    return SessionOut.of(session, history, made=made.get(session_id))


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(payload: SessionCreate, user: CurrentUser, db: DbSession):
    # Server-side refusal, not just a hidden menu entry.
    if payload.kind.value not in await settings_store.enabled_kinds():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="이 기능은 사용할 수 없습니다."
        )
    await _validate_session_links(
        db,
        user,
        payload.kind,
        project_id=payload.project_id,
        agent_id=payload.agent_id,
    )
    if payload.routing_mode in _AUTO_MODES and payload.kind is not SessionKind.chat:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="auto_routing_chat_only",
        )
    if payload.model == "auto":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="auto_is_not_a_model_id",
        )
    if payload.routing_mode in _AUTO_MODES:
        await _require_auto_quality_model(user, payload.model)
    session = ChatSession(
        user_id=user.id,
        kind=payload.kind,
        project_id=payload.project_id,
        agent_id=payload.agent_id,
        model=payload.model or "",
        routing_mode=payload.routing_mode,
        render_template_id=await _project_render_template(db, payload.project_id, payload.kind),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionOut.of(session, [])


async def _template_skills(
    db: AsyncSession,
    template: design_templates.DesignTemplate | None,
    kind: SessionKind,
    skills_event: dict | None,
) -> tuple[list[str], dict | None]:
    """`(context blocks, skills_applied event)` with the template's catalogue skills joined.

    Built-in org-visible rows only, by `catalog_key`; unknown keys are skipped.
    """
    if template is None or not template.skills:
        return [], skills_event
    rows = (
        await db.exec(
            select(Skill)
            .where(
                col(Skill.catalog_key).in_(list(template.skills)),
                Skill.source == SkillSource.built_in,
                Skill.visibility == SkillVisibility.org,
            )
            .order_by(col(Skill.id))
        )
    ).all()
    by_key = {row.catalog_key: row for row in rows}
    blocks: list[str] = []
    applied: list[dict] = []
    for key in template.skills:
        skill = by_key.get(key)
        if skill is None or (skill.kinds and kind.value not in skill.kinds):
            continue
        head = f"# 서식 스킬 — {skill.name}"
        body = skill.body.strip() or skill.description.strip()
        blocks.append(f"{head}\n{body}" if body else head)
        applied.append(
            {
                "id": skill.id,
                "name": skill.name,
                "catalogKey": skill.catalog_key,
                "estimatedTokens": skill.estimated_tokens,
                # Lets the screen mark template-activated skills.
                "fromTemplate": True,
            }
        )
    if not applied:
        return blocks, skills_event
    event = dict(skills_event or {"type": "skills_applied", "skills": []})
    event["skills"] = list(event.get("skills") or []) + applied
    # The browser requires `estimatedTokens` on the event.
    event["estimatedTokens"] = sum(
        int(skill.get("estimatedTokens") or 0) for skill in event["skills"]
    )
    return blocks, event


async def _project_render_template(
    db: DbSession, project_id: str | None, kind: SessionKind
) -> str | None:
    """The project's default template for this kind.

    Copied onto the session so a later project change does not affect it.
    """
    if not project_id:
        return None
    project = await db.get(Project, project_id)
    return design_templates.default_for(project.render_templates, kind) if project else None


def _resolved_template_id(requested: str | None, kind: SessionKind) -> str | None:
    """A rendering template id usable on this surface, or `None`.

    Unknown ids are refused, not dropped.
    """
    if not requested:
        return None
    chosen = design_templates.get(requested)
    if chosen is None or chosen.kind not in design_templates.HTML_KINDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="design_template_not_found"
        )
    if chosen.surface is not kind:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="design_template_surface_mismatch",
        )
    return chosen.id


@router.patch("/{session_id}", response_model=SessionOut)
async def patch_session(session_id: str, payload: SessionPatch, user: CurrentUser, db: DbSession):
    session = await _owned(db, user, session_id)
    changes = payload.model_dump(exclude_unset=True)
    if changes.get("model") == "auto":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="auto_is_not_a_model_id",
        )
    if changes.get("routing_mode") in _AUTO_MODES and session.kind is not SessionKind.chat:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="auto_routing_chat_only",
        )
    # A model-only patch switches to manual.
    if "model" in changes and "routing_mode" not in changes:
        changes["routing_mode"] = RoutingMode.manual
    # Validate the post-patch state.
    if changes.get("routing_mode", session.routing_mode) in _AUTO_MODES:
        await _require_auto_quality_model(user, changes.get("model", session.model))
    if "render_template_id" in changes:
        changes["render_template_id"] = _resolved_template_id(
            changes["render_template_id"], session.kind
        )
    if "project_id" in changes:
        await _validate_session_links(
            db,
            user,
            session.kind,
            project_id=changes["project_id"],
            agent_id=session.agent_id,
        )
    for field, value in changes.items():
        setattr(session, field, value)
    session.updated_at = utcnow()
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionOut.of(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, user: CurrentUser, db: DbSession):
    session = await _owned(db, user, session_id)
    await db.exec(delete(Message).where(Message.session_id == session.id))
    # Job rows reference the session without cascade.
    await db.exec(delete(Job).where(Job.session_id == session.id))
    await _delete_artifacts_of(db, [session.id])
    if session.index_key:
        await index_client.forget_collection(collection=session.index_key)
    await db.delete(session)
    await db.commit()


async def _delete_artifacts_of(db: DbSession, session_ids: list[str]) -> int:
    """Deletes what these conversations produced and returns how many artifacts went.

    A conversation and its artifacts are one record: deleting the one deletes the other,
    so nothing outlives the request that made it. Versions go first; shares cascade.
    """
    made = (
        await db.exec(select(Artifact.id).where(col(Artifact.session_id).in_(session_ids)))
    ).all()
    if made:
        await db.exec(delete(ArtifactVersion).where(col(ArtifactVersion.artifact_id).in_(made)))
        await db.exec(delete(Artifact).where(col(Artifact.id).in_(made)))
    return len(made)


def _record_media(
    db: DbSession,
    session: ChatSession,
    prompt: str,
    made: list[Artifact],
    *,
    model: str = "",
    credits: int = 0,
    failed: bool = False,
) -> None:
    """Writes a picture or clip turn as an ordinary user/assistant message pair.

    Nothing made leaves the prompt unanswered; a partial batch keeps what arrived.
    """
    if not session.title:
        session.title = chat_service.provisional_title(prompt)
    db.add(chat_service.media_prompt(session.id, prompt, unanswered=failed and not made))
    if made:
        db.add(
            chat_service.media_answer(
                session.id,
                [artifact.id for artifact in made],
                model=model,
                credits=credits,
                partial=failed,
            )
        )
        # Newest result for the panel, except on document surfaces where
        # `artifact_id` is the report or deck.
        if session.kind not in (SessionKind.report, SessionKind.slides):
            session.artifact_id = made[-1].id
    session.updated_at = utcnow()
    db.add(session)


@router.post("/{session_id}/figure-suggestion", response_model=FigureSuggestion)
async def suggest_figure(
    session_id: str, payload: FigureSuggestRequest, user: CurrentUser, db: DbSession
):
    """Proposes a caption and prompt for a figure. Nothing is drawn or charged."""
    session = await _owned(db, user, session_id)
    catalogue = await model_service.list_models()
    model = model_service.find(catalogue["models"], session.model or "")
    if model is None or "chat" not in model.get("kinds", []):
        usable = sorted(
            (m for m in catalogue["models"] if "chat" in m.get("kinds", [])),
            key=model_service.fallback_order,
        )
        if not usable:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no_chat_models"
            )
        model = usable[0]
    await litellm_service.ensure_key(user)
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    _, api_key = await litellm_service.credentials_for(user)
    figure = await figures.suggest(
        title=payload.title or session.title or "",
        about=payload.about,
        context=payload.context,
        model=str(model["id"]),
        api_key=api_key,
        look=payload.visual_style,
    )
    # No suggestion is not an error; the box stays empty.
    if figure is None:
        return FigureSuggestion(caption="", prompt="")
    return FigureSuggestion(
        caption=figure.caption,
        prompt=figure.prompt,
        template_id=figure.template_id,
        figure=figure.figure,
        description=figure.description,
        style=figure.style,
    )


@router.post("/{session_id}/images", response_model=list[ArtifactOut])
async def generate_images(session_id: str, payload: ImageRequest, user: CurrentUser, db: DbSession):
    """Makes pictures and stores them as artifacts.

    Synchronous, one image per upstream call. Charged from reported usage, not an estimate.
    """
    session = await _owned(db, user, session_id)
    catalogue = await model_service.list_models()
    if payload.style == "차트":
        # A chart is drawn from code, not painted: see `chart_code`.
        return await _draw_chart(session, payload, user, db, catalogue)
    model = model_service.find(catalogue["models"], payload.model or session.model or "")
    if model is None or "image" not in model["kinds"]:
        usable = sorted(
            (m for m in catalogue["models"] if "image" in m["kinds"]),
            key=model_service.fallback_order,
        )
        if not usable:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no_image_models"
            )
        model = usable[0]
    if not has_headroom(user, model):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="insufficient_credits"
        )

    await litellm_service.ensure_key(user)
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    base_url, api_key = await litellm_service.credentials_for(user)
    picture_template = design_templates.get(payload.template_id)
    if payload.template_id and (picture_template is None or picture_template.kind != "image"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="design_template_not_found"
        )
    # A language model expands the prompt unless the person sends back an edited one (`raw`).
    planned = payload.prompt
    if not payload.raw and payload.style != "없음":
        planner = model_service.find(catalogue["models"], session.model or "")
        if planner is None or "chat" not in planner.get("kinds", []):
            chat_models = sorted(
                (m for m in catalogue["models"] if "chat" in m.get("kinds", [])),
                key=model_service.fallback_order,
            )
            planner = chat_models[0] if chat_models else None
        if planner is not None:
            planned, _ = await imagegen.plan(
                payload.prompt,
                style=payload.style,
                # A wordless 서식 (poster, cover, banner) outranks the label chip: the
                # planner would otherwise write a title and captions the suffix then forbids.
                labels="none" if picture_template and picture_template.wordless else payload.labels,
                figure=payload.figure,
                model=str(planner["id"]),
                api_key=api_key,
                template=picture_template.prompt_suffix if picture_template else "",
            )
    composed = imagegen.compose_prompt(
        planned,
        aspect=payload.aspect,
        style=payload.style,
        template=picture_template.prompt_suffix if picture_template else "",
        design=design_service.image_clause(await design_for(db, user, session)),
        # A planned prompt already says it is one figure with no text.
        figure=payload.figure and planned == payload.prompt,
        square_only=not imagegen.honours_aspect(str(model["id"])),
    )

    made: list[Artifact] = []
    charged = 0
    failure: str | None = None
    for _ in range(payload.count):
        try:
            image = await imagegen.generate(
                base_url=base_url,
                api_key=api_key,
                model=model["id"],
                prompt=composed,
                aspect=payload.aspect,
            )
        except imagegen.ImageError as exc:
            # Images made before the failure are kept and billed.
            failure = str(exc)
            break

        file_id, key = imagegen.store(user.id, image)
        db.add(
            StoredFile(
                id=file_id,
                user_id=user.id,
                session_id=session.id,
                name=f"{payload.prompt[:40] or 'image'}.png",
                mime=image.mime,
                size=len(image.data),
                storage_key=key,
                tokens=0,
            )
        )
        artifact = Artifact(
            user_id=user.id,
            session_id=session.id,
            project_id=session.project_id,
            kind=ArtifactKind.image,
            title=payload.prompt[:200] or "이미지",
            data={
                "kind": "image",
                "jobId": None,
                # Prompt as typed, without the appended aspect and style phrases.
                "prompt": payload.prompt,
                "aspect": payload.aspect,
                # Requested ratio beside the delivered one; the two can differ.
                "actualAspect": image.aspect,
                "width": image.width,
                "height": image.height,
                "style": payload.style,
                "labels": payload.labels,
                # What the model was sent; editable and resendable as `raw`.
                "composedPrompt": composed,
                "seed": 0,
                "model": model["id"],
                "src": f"{settings.api_prefix}/files/{file_id}/content",
            },
        )
        db.add(artifact)
        made.append(artifact)
        charged += charge_for_tokens(model, image.input_tokens, image.output_tokens)

    if charged:
        settle(
            db,
            user,
            charged,
            reason="image.generate",
            session_id=session.id,
            model=model["id"],
            surface=session.kind.value,
        )
    _record_media(
        db,
        session,
        payload.prompt,
        made,
        model=model["id"],
        credits=charged,
        failed=failure is not None,
    )
    await db.commit()
    for artifact in made:
        await db.refresh(artifact)

    if not made:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=failure or "image_failed"
        )
    return [ArtifactOut.of(a) for a in made]


async def _draw_chart(
    session: ChatSession, payload: ImageRequest, user: User, db: DbSession, catalogue: dict
) -> list[ArtifactOut]:
    """The 차트 style: matplotlib code from a language model, run in the sandbox.

    Billed as the language model's tokens; the code is stored as `composedPrompt` for re-runs.
    """
    writer = model_service.find(catalogue["models"], session.model or "")
    if writer is None or "chat" not in writer.get("kinds", []):
        chat_models = sorted(
            (m for m in catalogue["models"] if "chat" in m.get("kinds", [])),
            key=model_service.fallback_order,
        )
        if not chat_models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no_chat_models"
            )
        writer = chat_models[0]
    if not has_headroom(user, writer):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="insufficient_credits"
        )
    await litellm_service.ensure_key(user)
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    _, api_key = await litellm_service.credentials_for(user)
    try:
        chart = await chart_code.draw(
            payload.prompt,
            aspect=payload.aspect,
            model=str(writer["id"]),
            api_key=api_key,
            code=payload.prompt if payload.raw else None,
        )
    except chart_code.ChartError as exc:
        _record_media(db, session, payload.prompt, [], model=str(writer["id"]), failed=True)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    image = imagegen.GeneratedImage(
        data=chart.png,
        mime="image/png",
        input_tokens=chart.input_tokens,
        output_tokens=chart.output_tokens,
    )
    image.width, image.height = imagegen._measure(chart.png)
    file_id, key = imagegen.store(user.id, image)
    db.add(
        StoredFile(
            id=file_id,
            user_id=user.id,
            session_id=session.id,
            name=f"{payload.prompt[:40] or 'chart'}.png",
            mime="image/png",
            size=len(chart.png),
            storage_key=key,
            tokens=0,
        )
    )
    artifact = Artifact(
        user_id=user.id,
        session_id=session.id,
        project_id=session.project_id,
        kind=ArtifactKind.image,
        title=(payload.prompt if not payload.raw else "차트")[:200] or "차트",
        data={
            "kind": "image",
            "jobId": None,
            "prompt": payload.prompt if not payload.raw else "차트",
            "aspect": payload.aspect,
            "actualAspect": image.aspect,
            "width": image.width,
            "height": image.height,
            "style": "차트",
            "labels": payload.labels,
            "engine": "matplotlib",
            "composedPrompt": chart.code,
            "seed": 0,
            "model": str(writer["id"]),
            "src": f"{settings.api_prefix}/files/{file_id}/content",
        },
    )
    db.add(artifact)
    charged = charge_for_tokens(writer, chart.input_tokens, chart.output_tokens)
    if charged:
        settle(
            db,
            user,
            charged,
            reason="image.chart",
            session_id=session.id,
            model=str(writer["id"]),
            surface=session.kind.value,
        )
    _record_media(db, session, payload.prompt, [artifact], model=str(writer["id"]), credits=charged)
    await db.commit()
    await db.refresh(artifact)
    return [ArtifactOut.of(artifact)]


@router.post("/{session_id}/diagrams", response_model=DiagramOut)
async def write_diagram(session_id: str, payload: DiagramRequest, user: CurrentUser, db: DbSession):
    """Writes a labelled figure as mermaid source; the client renders it."""
    session = await _owned(db, user, session_id)
    catalogue = await model_service.list_models()
    usable = sorted(
        (m for m in catalogue["models"] if "chat" in m["kinds"]),
        key=model_service.fallback_order,
    )
    # `session.model` is a picture model on this surface; a chat model is needed.
    asked = model_service.find(catalogue["models"], payload.model or session.model or "")
    model = asked if asked and "chat" in asked["kinds"] else (usable[0] if usable else None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no_models_available"
        )
    if not has_headroom(user, model):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="insufficient_credits"
        )
    await litellm_service.ensure_key(user)
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    _, api_key = await litellm_service.credentials_for(user)

    try:
        source, caption, usage = await diagram_service.draw(
            description=payload.description,
            figure=payload.figure,
            model=model["id"],
            api_key=api_key,
            language=payload.language,
            broken=payload.broken,
            error=payload.error,
        )
    except Exception as exc:  # noqa: BLE001 — the caller gets a reason, not a 500
        log.warning("diagram failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="diagram_failed"
        ) from exc

    charged = charge_for_tokens(
        model, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    )
    if charged:
        settle(
            db,
            user,
            charged,
            reason="image.diagram",
            session_id=session.id,
            model=model["id"],
            surface=session.kind.value,
        )
        await db.commit()
    return DiagramOut(source=source, caption=caption, model=model["id"], credits=charged)


@router.post("/{session_id}/diagrams/store", response_model=ArtifactOut)
async def store_diagram_image(
    session_id: str, payload: DiagramStore, user: CurrentUser, db: DbSession
):
    """Stores a client-rendered figure PNG as an image artifact with its mermaid source."""
    session = await _owned(db, user, session_id)
    try:
        blob = base64.b64decode(payload.png, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="bad_png"
        ) from exc
    if len(blob) > 8_000_000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="png_too_large"
        )
    file_id = uuid.uuid4().hex
    key = file_service.write_blob(user.id, file_id, "figure.png", blob)
    title = (payload.title or payload.caption or "도식")[:200]
    db.add(
        StoredFile(
            id=file_id,
            user_id=user.id,
            session_id=session.id,
            name=f"{title[:40]}.png",
            mime="image/png",
            size=len(blob),
            storage_key=key,
            tokens=0,
        )
    )
    artifact = Artifact(
        user_id=user.id,
        session_id=session.id,
        project_id=session.project_id,
        kind=ArtifactKind.image,
        title=title,
        data={
            "kind": "image",
            "jobId": None,
            "prompt": payload.description,
            "aspect": "16:9",
            "actualAspect": f"{payload.width}:{payload.height}",
            "width": payload.width,
            "height": payload.height,
            "style": "figure",
            "seed": 0,
            "model": payload.model,
            "src": f"{settings.api_prefix}/files/{file_id}/content",
            # The source is the artifact; the PNG is one rendering.
            "figure": payload.figure,
            "source": payload.source,
            "caption": payload.caption,
        },
    )
    db.add(artifact)
    _record_media(db, session, payload.description, [artifact], model=payload.model, credits=0)
    await db.commit()
    await db.refresh(artifact)
    return ArtifactOut.of(artifact)


@router.post("/{session_id}/audio", response_model=ArtifactOut)
async def generate_audio(session_id: str, payload: AudioRequest, user: CurrentUser, db: DbSession):
    """Makes one sound clip and stores it as an artifact.

    Speech and music are separate model families.
    """
    session = await _owned(db, user, session_id)
    speech = payload.audio_kind == "narration"
    catalogue = await model_service.list_models()

    def _audio_models():
        return [m for m in catalogue["models"] if "av" in m["kinds"]]

    model = model_service.find(catalogue["models"], payload.model or "")
    # The model must match the requested kind; a mismatch is a proxy 400.
    wanted_id = "gpt-audio" if speech else "lyria"
    if model is None or "av" not in model["kinds"] or wanted_id not in model["id"]:
        # Speech: OpenAI audio. Music: Lyria. Selected by id.
        candidates = [m for m in _audio_models() if wanted_id in m["id"]]
        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no_audio_models"
            )
        model = candidates[0]
    if not has_headroom(user, model):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="insufficient_credits"
        )

    await litellm_service.ensure_key(user)
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    base_url, api_key = await litellm_service.credentials_for(user)

    try:
        audio = await audiogen.generate(
            base_url=base_url,
            api_key=api_key,
            model=model["id"],
            prompt=payload.prompt,
            speech=speech,
            voice=payload.voice,
            seconds=payload.seconds,
        )
    except audiogen.AudioError as exc:
        # The prompt is recorded even when nothing came of it.
        _record_media(db, session, payload.prompt, [], failed=True)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    file_id, key = audiogen.store(user.id, audio)
    db.add(
        StoredFile(
            id=file_id,
            user_id=user.id,
            session_id=session.id,
            name=f"{payload.prompt[:40] or 'audio'}.{audio.extension}",
            mime=audio.mime,
            size=len(audio.data),
            storage_key=key,
            tokens=0,
        )
    )
    artifact = Artifact(
        user_id=user.id,
        session_id=session.id,
        project_id=session.project_id,
        kind=ArtifactKind.audio,
        title=payload.prompt[:200] or "오디오",
        data={
            "kind": "audio",
            "jobId": None,
            "prompt": payload.prompt,
            "audioKind": payload.audio_kind,
            "voice": payload.voice if speech else "",
            # Requested length beside the delivered one.
            "requestedSec": payload.seconds,
            "durationSec": audiogen.duration_seconds(audio),
            "model": model["id"],
            "transcript": audio.transcript,
            # Flat placeholder waveform; the real one would mean decoding the clip.
            "waveform": [],
            "src": f"{settings.api_prefix}/files/{file_id}/content",
        },
    )
    db.add(artifact)

    # Lyria: flat $0.04 per song, four reported tokens. Token billing reads as free.
    charged = max(
        charge_for_tokens(model, audio.input_tokens, audio.output_tokens),
        int(model.get("creditPerCall") or 0),
    )
    if charged:
        settle(
            db,
            user,
            charged,
            reason="audio.generate",
            session_id=session.id,
            model=model["id"],
            surface=session.kind.value,
        )
    _record_media(db, session, payload.prompt, [artifact], model=model["id"], credits=charged)
    await db.commit()
    await db.refresh(artifact)
    return ArtifactOut.of(artifact)


@router.post("/delete")
async def delete_sessions(payload: SessionBulkDelete, user: CurrentUser, db: DbSession):
    """Deletes many conversations in one request. `all` is separate from an id list."""
    query = select(ChatSession).where(ChatSession.user_id == user.id)
    if not payload.all:
        if not payload.ids:
            return {"deleted": 0}
        query = query.where(col(ChatSession.id).in_(payload.ids))
    rows = (await db.exec(query)).all()
    if not rows:
        return {"deleted": 0}

    ids = [row.id for row in rows]
    await db.exec(delete(Message).where(col(Message.session_id).in_(ids)))
    # Job rows reference the session without cascade.
    await db.exec(delete(Job).where(col(Job.session_id).in_(ids)))

    artifacts_deleted = await _delete_artifacts_of(db, ids)
    for row in rows:
        if row.index_key:
            await index_client.forget_collection(collection=row.index_key)

    await db.exec(delete(ChatSession).where(col(ChatSession.id).in_(ids)))
    await db.commit()
    return {"deleted": len(ids), "artifactsDeleted": artifacts_deleted}


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(session_id: str, user: CurrentUser, db: DbSession):
    await _owned(db, user, session_id)
    return [MessageOut.of(m) for m in await _history(db, session_id)]


def _regeneration_summary(request: str) -> str:
    """Distinct version label derived from the regeneration request."""
    # First line only; the conditions block `merge_answers` appends is noise here.
    said = " ".join(request.split("덧붙인 조건:")[0].split())[:80]
    return f"재생성 전 · {said}" if said else "재생성 전"


async def _store_document(
    db: AsyncSession,
    session: ChatSession,
    *,
    user_id: str,
    project_id: str | None,
    kind: ArtifactKind,
    title: str,
    data: dict,
    summary: str,
) -> str:
    """Store same-kind regeneration as a version; create a row for a new kind."""
    existing = await db.get(Artifact, session.artifact_id) if session.artifact_id else None
    if (
        existing is not None
        and existing.kind is kind
        and existing.user_id == user_id
        and existing.session_id == session.id
    ):
        db.add(
            ArtifactVersion(
                artifact_id=existing.id,
                version=existing.version,
                data=existing.data,
                storage_key=existing.storage_key,
                summary=summary,
            )
        )
        existing.version += 1
        existing.title = title
        existing.data = data
        existing.updated_at = utcnow()
        db.add(existing)
        return existing.id

    artifact = Artifact(
        user_id=user_id,
        session_id=session.id,
        project_id=project_id,
        kind=kind,
        title=title,
        data=data,
    )
    db.add(artifact)
    await db.flush()
    return artifact.id


async def _settle_plan_turn(
    *,
    session_id: str,
    user_id: str,
    request: str,
    attachments: list[str],
    answers: dict[str, str],
    model: dict,
    outline_model: dict | None,
    usage: dict[str, int],
    proposal: dict | None,
    questions: list[dict] | None,
    #: The figure question, asked only after the outline is approved.
    figures: dict | None = None,
) -> None:
    """Stores a turn that planned or asked, and charges for the planning call.

    No artifact is created and `session.artifact_id` is untouched.
    """
    credits = charge_for_tokens(model, usage.get("inputTokens", 0), usage.get("outputTokens", 0))
    credits += charge_for_tokens(
        outline_model or model,
        usage.get("outlineInputTokens", 0),
        usage.get("outlineOutputTokens", 0),
    )
    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        user = await db.get(User, user_id)
        if session is None or user is None:
            return
        stage = "clarify" if questions else "figures" if figures else "outline"
        session.pending = {
            "stage": stage,
            "request": request,
            "attachments": attachments,
            "answers": answers,
            **({"questions": questions} if questions else {}),
            **({"plan": proposal} if proposal else {}),
            # The approved outline is carried through the figure question.
            **(figures or {}),
        }
        db.add(
            Message(
                session_id=session_id,
                role=Role.assistant,
                content=(
                    "시작하기 전에 확인할 것이 있습니다."
                    if questions
                    else "그림을 넣을지 확인해 주세요."
                    if figures
                    else "이렇게 구성하려고 합니다. 확인해 주세요."
                ),
                usage={**usage, "credits": credits},
                model=model["id"],
            )
        )
        settle(
            db,
            user,
            credits,
            reason="document.plan",
            session_id=session_id,
            model=(outline_model or model)["id"],
            # The reason does not name the surface; the session does.
            surface=session.kind.value,
        )
        session.updated_at = utcnow()
        db.add(session)
        await db.commit()


#: Above any surface's real outline size; refuses a bad payload, never bounds a real one.
_MAX_PLANNED = 60


def _edited_plan(sent: dict | None, stored: dict) -> dict:
    """The stored outline with the person's edits folded in.

    Titles and order come from the browser; layouts and every other key from the proposal.
    """
    if not isinstance(sent, dict):
        return stored
    out = dict(stored)
    for key in ("sections", "slides", "blocks"):
        items = sent.get(key)
        if key not in stored or not isinstance(items, list) or not items:
            continue
        if len(items) > _MAX_PLANNED:
            continue
        # Layouts must be ones the proposal already used.
        if all(isinstance(row, str) for row in items):
            kept = [row.strip()[:200] for row in items if row.strip()]
        else:
            layouts = {
                str(row.get("layout") or "") for row in stored.get(key, []) if isinstance(row, dict)
            }
            kept = []
            for row in items:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "").strip()[:200]
                layout = str(row.get("layout") or "")
                if not title or layout not in layouts:
                    continue
                kept.append({"title": title, "layout": layout})
        if kept:
            out[key] = kept
    if title := str(sent.get("title") or "").strip():
        out["title"] = title[:200]
    visual_style = str(sent.get("visualStyle") or "").strip()
    if visual_style in design_service.VISUAL_STYLES:
        out["visualStyle"] = visual_style
    density = str(sent.get("density") or "").strip()
    if density in ("speaker", "reading"):
        out["density"] = density
    return out


async def _ask_before_writing(
    db: DbSession,
    session: ChatSession,
    *,
    request: str,
    typed: str,
    attachments: list[str],
    questions: list[grounding.Question],
) -> JSONResponse:
    """Persist a clarification question without generating a document."""
    session.pending = {
        "stage": "clarify",
        "request": request,
        "attachments": attachments,
        "questions": [q.wire() for q in questions],
        "answers": {},
    }
    session.updated_at = utcnow()
    db.add(session)
    db.add(
        Message(
            session_id=session.id,
            role=Role.user,
            content=typed,
            attachments=None,
        )
    )
    said = "시작하기 전에 확인할 것이 있습니다."
    db.add(
        Message(
            session_id=session.id,
            role=Role.assistant,
            content=said,
        )
    )
    await db.commit()
    # The sentence travels with the questions so the in-flight bubble is not empty.
    return JSONResponse({"pending": session.pending, "message": said})


def _plans_first(session: ChatSession) -> bool:
    """Whether this surface offers an outline before writing (report and slides)."""
    return session.kind in (SessionKind.report, SessionKind.slides)


@router.post("/{session_id}/messages")
async def send_message(
    session_id: str, payload: SendMessage, request: Request, user: CurrentUser, db: DbSession
):
    session = await _owned(db, user, session_id)

    #: The user message being rerun; its stored words and attachments replace the client's.
    retry_of: Message | None = None
    if payload.retry_of:
        retry_of = await db.get(Message, payload.retry_of)
        if retry_of is None or retry_of.session_id != session.id or retry_of.role is not Role.user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="retry_target_not_found"
            )
        payload = payload.model_copy(
            update={
                "content": retry_of.content,
                "attachments": payload.attachments
                or [a["id"] for a in (retry_of.attachments or []) if a.get("id")]
                or None,
            }
        )

    # A document surface plans before it writes; `session.pending` holds the
    # half-finished turn, and a message while one is pending is a note on it.
    pending = dict(session.pending or {}) if _plans_first(session) else {}
    #: Set only via approval; a plain message never carries one.
    approved_plan: dict | None = None
    #: Set by the figure card; `None` until either button is pressed.
    approved_figures: list[dict] | None = None
    #: A message working on the finished document rather than asking for a new
    #: one. `services.revise` decides what it lands on.
    revising = bool(
        _plans_first(session)
        and session.artifact_id
        and not pending
        and not payload.approve
        and payload.answers is None
        and not revise.obviously_new(payload.content)
    )
    focus = ""
    #: What the person typed, as opposed to the merged request the model gets.
    typed_content = payload.content
    #: "있는 자료로 진행" sends an empty answers object; it tells the next pass not to ask again.
    proceed_as_is = bool(
        pending and not payload.approve and payload.answers is not None and not payload.answers
    )
    if pending:
        answers = {**(pending.get("answers") or {}), **(payload.answers or {})}
        pending["answers"] = answers
        focus = grounding.focus_terms(answers)
        # Approval is the one path that does not plan again.
        if payload.approve and pending.get("plan"):
            approved_plan = _edited_plan(payload.plan, dict(pending["plan"]))
        # Pictures agreed to, or `[]` when declined.
        if pending.get("stage") == "figures" and payload.include_figures is not None:
            approved_figures = list(pending.get("figures") or []) if payload.include_figures else []
        payload = payload.model_copy(
            update={
                "content": grounding.merge_answers(
                    str(pending.get("request") or ""),
                    # An approval is not a condition on the request.
                    answers if payload.approve else {**answers, "_note": typed_content},
                ),
                # A reply carries the original request's attachments.
                "attachments": payload.attachments or list(pending.get("attachments") or []),
            }
        )

    if session.kind not in (SessionKind.chat, SessionKind.report, SessionKind.slides):
        # Image and a/v are jobs with their own endpoints, not this path.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="surface_not_implemented"
        )
    auto_turn = bool(
        session.kind is SessionKind.chat
        and session.routing_mode in _AUTO_MODES
        and payload.model is None
    )

    # Refused before any write, never silently dropped.
    if payload.render_template_id is not None:
        session.render_template_id = _resolved_template_id(payload.render_template_id, session.kind)

    # Policy before any write, billing entry, virtual-key issue or model call.
    policy = await _require_egress_policy()
    content = payload.content
    if policy.intent_filter and (hit := governance.blocked_by(content, policy.blocked_categories)):
        await _audit_policy(user, request, "filter.blocked", hit)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"blocked_category:{hit}",
        )
    catalogue = await model_service.list_models_for_egress()
    catalogue_models = catalogue["models"]
    usable = _allowed_models(user, catalogue_models, kind=session.kind.value)
    # Model precedence: turn override, then session, then agent.
    try:
        agent_model, agent_tools, agent_temperature = await agent_settings(db, user, session)
        # 0.4 unless the agent sets its own: keeps the house style consistent.
        if agent_temperature is None:
            agent_temperature = 0.4
    except WorkspaceContextError as exc:
        _raise_workspace_error(exc)
    model_id = session.model if auto_turn else payload.model or session.model or agent_model
    model = model_service.find(catalogue_models, model_id) if model_id else None
    if payload.model and (model is None or session.kind.value not in model.get("kinds", [])):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_unavailable",
        )
    allowed_ids = set(user.allowed_models or [])
    if payload.model and allowed_ids and payload.model not in allowed_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="model_not_allowed",
        )
    if auto_turn and _find_auto_quality_model(usable, model_id) is None:
        # Auto is a ceiling, not permission to substitute a cheaper model.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="auto_quality_model_required",
        )
    # Kept so routing metadata and the transcript can name the substitute.
    revoked_model = model if model is not None and model not in usable else None
    if model not in usable:
        model = None
    if model is None:
        # Fallback stays inside the allowlist.
        usable = sorted(usable, key=model_service.fallback_order)
        if not usable:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no_models_available"
            )
        model = usable[0]
    requested_model = model

    history = await _history(db, session_id)
    #: What a retry replaces: the failed reply, if any. Neither it nor the
    #: question stays in the model's history.
    superseded: list[Message] = []
    if retry_of is not None:
        at = next((i for i, m in enumerate(history) if m.id == retry_of.id), None)
        if at is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="retry_target_not_found"
            )
        # Only the latest question may be rerun.
        if any(m.role is Role.user for m in history[at + 1 :]):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="retry_not_latest")
        superseded = history[at + 1 :]
        history = history[:at]

    # Ownership resolved before privacy assembly.
    rows, attachment_meta = await _owned_attachments(db, user, payload.attachments)

    stored_content = content
    outbound_history = [message.content for message in history]
    privacy_resolution: _PrivacyResolution | None = None
    tools: list[Tool] = []
    tool_definitions: list[dict[str, Any]] = []
    candidate_tools: list[Tool] = []
    strict_tools: list[Tool] = []

    # Tool definitions are outbound prompt data, so they are built before the
    # privacy decision.
    agent_row: WorkspaceAgent | None = None
    shelf: list[tuple[str, str, str | None]] = []
    shelf_key = ""
    if session.agent_id:
        agent_row = await db.get(WorkspaceAgent, session.agent_id)
    if session.kind is SessionKind.chat and requested_model.get("supportsTools"):
        shelf, shelf_key = await _knowledge_shelf(db, user, session, agent_row)

    requested_is_strict = _strict_model(requested_model)
    strict_candidate_available = (
        requested_is_strict
        or any(
            model.get("id") in set(policy.privacy_safe_model_ids or []) and _strict_model(model)
            for model in catalogue_models
        )
        or (
            auto_turn
            and any(
                model.get("id") in set(policy.adaptive_economy_model_ids or [])
                and _strict_model(model)
                for model in catalogue_models
            )
        )
    )
    # A sentence can request search as the toggle does; resolved before tools are built.
    explicit_web_search = requests_web_search(content)
    effective_web_search = payload.web_search or explicit_web_search
    # An agent whose allowlist leaves web search out chose that on purpose. The toggle
    # is moot for it, and a 「웹 검색 없이 답합니다」 preamble on every answer would
    # only be noise about a tool the agent was never meant to have.
    if session.agent_id and agent_tools is not None and "web_search" not in agent_tools:
        effective_web_search = False
    if session.kind is SessionKind.chat and requested_model.get("supportsTools"):
        if not requested_is_strict:
            candidate_tools = sorted(
                await build_tools(
                    db,
                    user,
                    web_search=effective_web_search,
                    allowed=agent_tools,
                    knowledge=shelf,
                    knowledge_collection=shelf_key,
                ),
                key=lambda tool: (tool.name, tool.source),
            )
        if strict_candidate_available:
            strict_tools = sorted(
                _strict_local_tools(
                    await build_tools(
                        db,
                        user,
                        web_search=False,
                        allowed=agent_tools,
                        knowledge=shelf,
                        knowledge_collection="",
                        include_connectors=False,
                        strict_local=True,
                    )
                ),
                key=lambda tool: (tool.name, tool.source),
            )

    requested_tools = strict_tools if requested_is_strict else candidate_tools
    try:
        workspace = await assemble(
            db,
            user,
            session,
            attachment_ids=payload.attachments,
            # Chat re-assembles below against the model privacy settles on.
            vision=reads_pictures(requested_model),
            file_budget=file_budget(requested_model),
            activated_skill_ids=payload.activated_skill_ids,
            starting_template_id=payload.starting_template_id,
            # Empty focus takes the head of the file.
            focus=focus or (content if session.kind is not SessionKind.chat else ""),
            question=content,
            # Report and deck writers do not run the chat tool loop.
            available_tool_names=(
                {tool.name for tool in requested_tools}
                if session.kind is SessionKind.chat
                else set()
            ),
        )
    except WorkspaceContextError as exc:
        _raise_workspace_error(exc)

    # Attachment shortfalls are known server-side, so ask before spending a planning call.
    if _plans_first(session) and not pending.get("answers"):
        short = grounding.file_shortfalls(workspace.attachments)
        questions = grounding.questions_for(short)
        # Only when no file arrived at all.
        if not questions and (gap := grounding.missing_attachment(content, workspace.attachments)):
            questions = [gap]
        if questions:
            return await _ask_before_writing(
                db,
                session,
                request=content,
                typed=typed_content,
                attachments=list(payload.attachments or []),
                questions=questions,
            )

    if session.kind is SessionKind.chat:
        privacy_sources = _privacy_sources(content, history, workspace.blocks)
        if requested_tools:
            privacy_sources["tool_definitions"] = _tool_definition_source(requested_tools)
        # Auto scans the full envelope before any classifier, key or model call.
        auto_preflight_findings = (
            governance.findings(privacy_sources, legacy=policy.pii_masking) if auto_turn else []
        )
        resolved = await _resolve_privacy(
            user=user,
            session=session,
            policy=policy,
            catalogue=catalogue_models,
            requested=[model],
            sources=privacy_sources,
            explicit_action=payload.privacy_action,
            decision_token=payload.privacy_decision_token,
        )
        if isinstance(resolved, JSONResponse):
            await _audit_policy(
                user,
                request,
                "privacy.decision_required",
                governance.DETECTOR_VERSION,
                metadata=_decision_audit_metadata(resolved, session.id),
            )
            return resolved
        privacy_resolution = resolved
        if revoked_model is not None:
            # `actualModelChanged` compares against the requested model.
            resolved.routing = {
                **resolved.routing,
                "requestedModels": [revoked_model["id"]],
            }
        if auto_turn and auto_preflight_findings:
            # Privacy owns this turn; the classifier never sees the envelope.
            cost_routing = _cost_routing(
                decision="bypassed",
                reason_code="privacy_detected",
                requested_model=requested_model,
                selected_model=resolved.models[0],
            )
            resolved.routing = {**resolved.routing, "costRouting": cost_routing}
        model = resolved.models[0]
        strict_local = resolved.strict_local
        # Privacy routing cannot add tool support.
        if requested_model.get("supportsTools") and model.get("supportsTools"):
            tools = strict_tools if strict_local else candidate_tools
        masker = governance.mask_legacy if policy.pii_masking else governance.mask
        if resolved.findings:
            # Findings are stored masked whatever the action.
            stored_content = masker(content)[0]
            if attachment_meta:
                # Filenames and errors are user content too.
                attachment_meta = [
                    {
                        **item,
                        "name": masker(str(item.get("name") or ""))[0],
                        "error": (
                            masker(str(item["error"]))[0]
                            if item.get("error")
                            else item.get("error")
                        ),
                    }
                    for item in attachment_meta
                ]
        if resolved.mask_outbound:
            content = masker(content)[0]
            outbound_history = _mask_list(outbound_history, legacy=policy.pii_masking)
            # Whole tools are dropped; schema keys cannot be rewritten safely.
            tools = _drop_sensitive_tool_definitions(
                tools,
                legacy=policy.pii_masking,
            )
        tool_definitions = openai_snapshot(tools)
    elif policy.pii_masking:
        # The decision flow is chat-only; report and slides always mask.
        masker = governance.mask_legacy
        content, masked = masker(content)
        stored_content = content
        if attachment_meta:
            # A filename or extraction error is user content.
            attachment_meta = _mask_text_tree(attachment_meta, masker)
        if masked:
            await _audit_policy(user, request, "pii.masked", f"{masked}건")

    # Re-resolve skills against the final tool snapshot.
    if session.kind is SessionKind.chat:
        try:
            workspace = await assemble(
                db,
                user,
                session,
                attachment_ids=payload.attachments,
                vision=reads_pictures(model),
                file_budget=file_budget(model),
                activated_skill_ids=payload.activated_skill_ids,
                starting_template_id=payload.starting_template_id,
                question=content,
                available_tool_names={tool.name for tool in tools},
            )
        except WorkspaceContextError as exc:
            _raise_workspace_error(exc)

    trusted_context = workspace.trusted
    untrusted_context = workspace.untrusted
    if privacy_resolution and privacy_resolution.mask_outbound:
        trusted_context = _mask_list(trusted_context, legacy=policy.pii_masking)
        untrusted_context = _mask_list(untrusted_context, legacy=policy.pii_masking)
    elif policy.pii_masking:
        # Always-mask covers the context blocks too.
        trusted_context = _mask_text_tree(trusted_context, governance.mask_legacy)
        untrusted_context = _mask_text_tree(untrusted_context, governance.mask_legacy)
    skills_event = workspace.skills_event()
    # Template skills join the same event as hand-activated ones.
    template_blocks, skills_event = await _template_skills(
        db, design_templates.get(session.render_template_id), session.kind, skills_event
    )
    if template_blocks:
        trusted_context = list(trusted_context) + template_blocks
    context_steps = _context_steps(workspace)
    if policy.pii_masking:
        # Skill names are user-controlled, like filenames.
        if skills_event:
            skills_event = _mask_text_tree(skills_event, governance.mask_legacy)
        # Same for memory names.
        context_steps = _mask_text_tree(context_steps, governance.mask_legacy)

    # Report and slides have no privacy resolution to carry a substitution.
    document_routing = (
        _substitution_routing(revoked_model, model)
        if privacy_resolution is None and revoked_model is not None
        else None
    )

    strict_local = bool(privacy_resolution and privacy_resolution.strict_local)
    wire_history = [
        {"role": message.role.value, "content": body}
        for message, body in zip(history, outbound_history, strict=True)
    ]
    wire_history.append({"role": "user", "content": content})
    messages = build_messages(
        session.kind,
        wire_history,
        with_tools=bool(tools),
        web_search=effective_web_search,
        # Whether a search tool survived, so the answer does not read as searched.
        web_search_available=any(t.name == "web_search" for t in tools),
        extra=trusted_context,
        untrusted_context=untrusted_context,
    )
    # After `build_messages`, never inside it — see `context.with_pictures`.
    messages = with_pictures(messages, [picture.uri for picture in workspace.pictures])

    if auto_turn and not auto_preflight_findings:
        unsupported = bool(
            payload.attachments
            or effective_web_search
            or payload.activated_skill_ids
            or payload.starting_template_id
            or session.agent_id
            or session.project_id
        )
        # Economy turns are tool-free; the context fit is checked on that envelope.
        economy_messages = build_messages(
            session.kind,
            wire_history,
            with_tools=False,
            web_search=False,
            extra=trusted_context,
            untrusted_context=untrusted_context,
        )
        routed_model, cost_routing = await _resolve_cost_routing(
            mode=session.routing_mode,
            db=db,
            user=user,
            policy=policy,
            catalogue=catalogue_models,
            quality_model=requested_model,
            classifier_messages=messages,
            classifier_tool_definitions=tool_definitions,
            context_tokens=adaptive_routing.estimated_context_tokens(
                [
                    json.dumps(
                        economy_messages,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ]
            ),
            unsupported_reason="unsupported_turn" if unsupported else None,
        )
        resolved.models = [routed_model]
        resolved.routing = _apply_effective_model(resolved.routing, routed_model)
        resolved.routing = {**resolved.routing, "costRouting": cost_routing}
        privacy_resolution = resolved
        model = routed_model
        # Only a downward route swaps to the tool-free envelope.
        if (
            cost_routing.get("decision") == "routed"
            and session.routing_mode != RoutingMode.auto_quality
        ):
            tools = []
            tool_definitions = []
            messages = economy_messages
        strict_local = resolved.strict_local

    if not has_headroom(user, model):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="insufficient_credits"
        )

    for stored in rows:
        stored.session_id = session.id
        db.add(stored)

    if retry_of is not None:
        # Reuse the question row; drop the failed reply and clear its failure mark.
        for row in superseded:
            await db.delete(row)
        user_message = retry_of
        user_message.failure = None
        user_message.routing = (
            privacy_resolution.routing if privacy_resolution else document_routing
        )
    else:
        user_message = Message(
            session_id=session.id,
            role=Role.user,
            # What they typed, not the merged request.
            content=typed_content if pending else stored_content,
            attachments=attachment_meta,
            routing=privacy_resolution.routing if privacy_resolution else document_routing,
            started_from=workspace.started_from,
        )
    db.add(user_message)
    # Turn overrides are not written back; an Auto session's model is its ceiling.
    if session.routing_mode not in _AUTO_MODES or payload.model is None:
        # A substitute is for this turn only.
        if revoked_model is None:
            session.model = requested_model["id"]
    session.updated_at = utcnow()
    if not session.title:
        # Provisional title, replaced once the first turn completes.
        session.title = chat_service.provisional_title(stored_content)
    db.add(session)
    privacy_audit_id: str | None = None
    if privacy_resolution and privacy_resolution.findings:
        privacy_audit = AuditEvent(
            actor_id=user.id,
            action=f"privacy.{privacy_resolution.action}",
            target=session.id,
            detail=governance.DETECTOR_VERSION,
            event_metadata={
                **governance.finding_metadata(privacy_resolution.findings),
                **privacy_resolution.routing,
            },
            ip=client_ip(request),
            severity="warn",
        )
        privacy_audit_id = privacy_audit.id
        db.add(privacy_audit)
    routing_audit_id: str | None = None
    cost_route = (
        privacy_resolution.routing.get("costRouting")
        if privacy_resolution and privacy_resolution.routing
        else None
    )
    if cost_route:
        routing_audit = AuditEvent(
            actor_id=user.id,
            action="routing.auto",
            target=session.id,
            detail=adaptive_routing.CLASSIFIER_VERSION,
            event_metadata=dict(cost_route),
            ip=client_ip(request),
        )
        routing_audit_id = routing_audit.id
        db.add(routing_audit)
    if agent_row is not None:
        agent_row.runs += 1
        db.add(agent_row)
    await db.commit()

    # Also issues the key to an account provisioned during a proxy outage.
    await litellm_service.ensure_key(user)
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    _, api_key = await litellm_service.credentials_for(user)

    is_first_turn = len(history) == 0

    # The planner is bound by the same allowlist, surface and boundary as the writer.
    outline_model = _planner_model(
        policy.outline_model_id,
        user=user,
        catalogue=catalogue_models,
        kind=session.kind.value,
        writer=model,
        strict_local=strict_local,
    )

    #: The image default draws a document's figures; none means no figure card.
    image_model = next(
        (
            m
            for m in _allowed_models(user, catalogue_models, kind="image")
            if "image" in (m.get("kinds") or [])
        ),
        None,
    )

    render_template = design_templates.get(session.render_template_id)
    # Document and deck templates go through the report and deck writers; only
    # other template kinds use the block writer.
    if render_template is not None and render_template.kind not in ("document", "deck"):
        return StreamingResponse(
            _survive_disconnect(
                _run_page(
                    may_ask=not proceed_as_is,
                    user_id=user.id,
                    api_key=api_key,
                    session_id=session.id,
                    # `None` plans and offers again.
                    approved_plan=approved_plan,
                    attachments=list(payload.attachments or []),
                    answers=dict(pending.get("answers") or {}),
                    model=model,
                    request=content,
                    project_id=session.project_id,
                    routing=document_routing,
                    template=render_template,
                    trusted_context=trusted_context,
                    untrusted_context=untrusted_context,
                    design_tokens=workspace.design_tokens,
                    skills_event=skills_event,
                    context_steps=context_steps,
                    outline_model=outline_model,
                    project_sources=[
                        {
                            "id": item.id,
                            "name": item.name,
                            "state": item.state,
                            "sourceUrl": item.source_url,
                            "locations": list(item.locations),
                        }
                        for item in workspace.knowledge
                    ],
                    # No network on a strict-local route.
                    web_search=effective_web_search and not strict_local,
                )
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # See `revising` above.
    if revising and session.kind in (SessionKind.report, SessionKind.slides):
        return StreamingResponse(
            _survive_disconnect(
                _revise_document(
                    user_id=user.id,
                    api_key=api_key,
                    session_id=session.id,
                    model=model,
                    instruction=content,
                    routing=document_routing,
                )
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if session.kind is SessionKind.report:
        return StreamingResponse(
            _survive_disconnect(
                _run_report(
                    may_ask=not proceed_as_is,
                    figures_plan=approved_figures,
                    image_model=image_model,
                    template=render_template,
                    user_id=user.id,
                    api_key=api_key,
                    session_id=session.id,
                    # `None` plans and offers again.
                    approved_plan=approved_plan,
                    attachments=list(payload.attachments or []),
                    answers=dict(pending.get("answers") or {}),
                    model=model,
                    request=content,
                    project_id=session.project_id,
                    routing=document_routing,
                    trusted_context=trusted_context,
                    untrusted_context=untrusted_context,
                    design_tokens=workspace.design_tokens,
                    skills_event=skills_event,
                    context_steps=context_steps,
                    outline_model=outline_model,
                    project_sources=[
                        {
                            "id": item.id,
                            "name": item.name,
                            "state": item.state,
                            "sourceUrl": item.source_url,
                            "locations": list(item.locations),
                        }
                        for item in workspace.knowledge
                    ],
                    # No network on a strict-local route.
                    web_search=effective_web_search and not strict_local,
                )
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if session.kind is SessionKind.slides:
        return StreamingResponse(
            _survive_disconnect(
                _run_deck(
                    template=render_template,
                    may_ask=not proceed_as_is,
                    figures_plan=approved_figures,
                    image_model=image_model,
                    user_id=user.id,
                    api_key=api_key,
                    session_id=session.id,
                    # `None` plans and offers again.
                    approved_plan=approved_plan,
                    attachments=list(payload.attachments or []),
                    answers=dict(pending.get("answers") or {}),
                    model=model,
                    request=content,
                    project_id=session.project_id,
                    routing=document_routing,
                    trusted_context=trusted_context,
                    untrusted_context=untrusted_context,
                    design_tokens=workspace.design_tokens,
                    skills_event=skills_event,
                    context_steps=context_steps,
                    outline_model=outline_model,
                    # No network on a strict-local route.
                    web_search=effective_web_search and not strict_local,
                )
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return StreamingResponse(
        _survive_disconnect(
            _run_turn(
                user_id=user.id,
                api_key=api_key,
                auto_memory=Preferences.of(user).auto_memory,
                session_id=session.id,
                # How far a shared note reaches, and whose byline it carries.
                project_id=session.project_id or "",
                agent_name=(agent_row.name if agent_row else ""),
                model=model,
                messages=messages,
                tools=tools,
                tool_definitions=tool_definitions,
                # None leaves the upstream default.
                temperature=agent_temperature,
                first_user_message=stored_content,
                # A turn with no answer records the failure on this row.
                user_message_id=user_message.id,
                is_first_turn=is_first_turn,
                skills_event=skills_event,
                context_steps=context_steps,
                routing=privacy_resolution.routing if privacy_resolution else document_routing,
                quality_model=requested_model if cost_route else None,
                disable_fallbacks=bool(cost_route and cost_route.get("decision") == "routed"),
                # Model output can introduce new findings, so it is masked at rest too.
                mask_at_rest=policy.pii_masking or policy.external_data_guard,
                sanitize_tool_output=bool(
                    policy.pii_masking
                    or (
                        policy.external_data_guard
                        and privacy_resolution
                        and not privacy_resolution.strict_local
                    )
                ),
                legacy_masking=policy.pii_masking,
                protect_enrichment=policy.pii_masking or policy.external_data_guard,
                privacy_audit_id=privacy_audit_id,
                routing_audit_id=routing_audit_id,
                # The toggle forces the first hop only.
                force_tool=(
                    "web_search"
                    if effective_web_search and any(t.name == "web_search" for t in tools)
                    else None
                ),
            )
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx buffers SSE.
            "X-Accel-Buffering": "no",
        },
    )


#: Active cancellation signals grouped by session
_STOPPING: dict[str, set[asyncio.Event]] = {}

#: A key echoed in an upstream error body; the reason is shown on screen.
_SECRET_IN_REASON = re.compile(r"sk-[A-Za-z0-9_\-]+")


def _error_event(message: str, exc: BaseException | None = None) -> dict[str, Any]:
    """SSE error with a machine code and bounded, redacted upstream reason."""
    code, reason = "internal_error", ""
    if isinstance(exc, chat_service.ChatStreamError):
        head, _, tail = str(exc).partition(": ")
        code = head.strip() or "upstream_failed"
        reason = _SECRET_IN_REASON.sub("sk-…", tail.strip())[:200]
    event: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if reason:
        event["reason"] = reason
    return event


async def _until_stopped(
    events: AsyncIterator[dict[str, Any]], stopping: asyncio.Event
) -> AsyncIterator[dict[str, Any]]:
    """Race stream delivery against cancellation and close the upstream generator."""
    iterator = events.__aiter__()
    waiting = asyncio.ensure_future(stopping.wait())
    try:
        while not stopping.is_set():
            nxt = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({nxt, waiting}, return_when=asyncio.FIRST_COMPLETED)
            if nxt not in done:
                # Cancel and await the pending read before `aclose()`, or it
                # raises on a running generator.
                nxt.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await nxt
                return
            try:
                event = nxt.result()
            except StopAsyncIteration:
                return
            yield event
    finally:
        waiting.cancel()
        await iterator.aclose()


#: Strong references; a task with no reader is otherwise collectible mid-turn.
_DETACHED: set[asyncio.Task] = set()

#: Idle gap before an SSE comment is sent. Under the 60s proxy idle timeout.
HEARTBEAT_SEC = 15.0


async def _heartbeat(events: AsyncIterator[str]) -> AsyncIterator[str]:
    """Relays a stream, sending an SSE comment after `HEARTBEAT_SEC` of silence.

    The pending read is kept across heartbeats; `wait_for` could drop an event
    on a coincident timeout.
    """
    iterator = events.__aiter__()
    nxt = asyncio.ensure_future(anext(iterator))
    try:
        while True:
            done, _ = await asyncio.wait({nxt}, timeout=HEARTBEAT_SEC)
            if not done:
                yield ": keep-alive\n\n"
                continue
            try:
                event = nxt.result()
            except StopAsyncIteration:
                return
            yield event
            nxt = asyncio.ensure_future(anext(iterator))
    finally:
        if not nxt.done():
            nxt.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await nxt
        await iterator.aclose()


def _survive_disconnect(events: AsyncIterator[str]) -> AsyncIterator[str]:
    """A turn's response: finishes behind a departed reader and stays alive while silent."""
    return _heartbeat(_detached(events))


async def _detached(events: AsyncIterator[str]) -> AsyncIterator[str]:
    """Lets a turn finish when nobody is reading.

    The work runs in its own task; this relays its queue.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in events:
                await queue.put(event)
        except Exception:  # noqa: BLE001 — nobody is left to receive a raise
            log.exception("detached turn failed")
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    _DETACHED.add(task)
    task.add_done_callback(_DETACHED.discard)

    while (event := await queue.get()) is not None:
        yield event


async def _store_notes(
    db: AsyncSession,
    user_id: str,
    session_id: str,
    project_id: str,
    notes: list[dict],
) -> None:
    """Writes a turn's handoff notes as memories, scoped to the project or else this session.

    `key` dedupes within the scope.
    """
    scope = project_id or session_id
    for note in notes:
        name = note["key"]
        existing = (
            await db.exec(
                select(Memory).where(
                    Memory.user_id == user_id,
                    Memory.scope == scope,
                    Memory.name == name,
                )
            )
        ).first()
        # Byline on the description, not the body.
        author = note.get("author") or ""
        description = f"{note['title']} — {author}" if author else note["title"]
        if existing is not None:
            existing.description = description[:400]
            existing.body = note["body"]
            existing.updated_at = utcnow()
            db.add(existing)
            continue
        db.add(
            Memory(
                user_id=user_id,
                name=name,
                description=description[:400],
                # A finding, not something the person said about themselves.
                type=MemoryType.reference,
                body=note["body"],
                scope=scope,
            )
        )


async def _run_turn(
    *,
    user_id: str,
    api_key: str,
    auto_memory: bool,
    session_id: str,
    project_id: str = "",
    agent_name: str = "",
    model: dict,
    messages: list[dict],
    tools: list[Tool],
    tool_definitions: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    first_user_message: str,
    user_message_id: str | None = None,
    is_first_turn: bool,
    skills_event: dict | None = None,
    context_steps: list[dict] | None = None,
    routing: dict | None = None,
    quality_model: dict | None = None,
    disable_fallbacks: bool = False,
    mask_at_rest: bool = False,
    sanitize_tool_output: bool = False,
    legacy_masking: bool = False,
    protect_enrichment: bool = False,
    privacy_audit_id: str | None = None,
    routing_audit_id: str | None = None,
    #: A tool the first hop must call. See `agent.run_turn`.
    force_tool: str | None = None,
) -> AsyncIterator[str]:
    """Drives one assistant turn to completion and settles it.

    Own DB session: the request-scoped one closes when the route returns the
    StreamingResponse.
    """
    text_parts: list[str] = []
    steps: list[dict] = _prelude_steps(skills_event, context_steps)
    usage = {"inputTokens": 0, "outputTokens": 0}
    failed: str | None = None
    answer_id: str | None = None
    tool_output_masked = 0
    tool_output_findings: dict[tuple[str, str], int] = {}
    actual_model = model["id"]

    # Set by the stop button, not by a closed socket.
    stopping = asyncio.Event()
    # An earlier turn on this session is superseded.
    for earlier in _STOPPING.get(session_id, set()):
        earlier.set()
    _STOPPING.setdefault(session_id, set()).add(stopping)

    ctx = ToolContext(
        user_id=user_id,
        session_id=session_id,
        api_key=api_key,
        project_id=project_id,
        agent_name=agent_name,
        # The latest user message, not the conversation's opening line.
        request=next(
            (
                str(one.get("content") or "")
                for one in reversed(messages)
                if one.get("role") == "user"
            ),
            "",
        ),
    )
    masker = governance.mask_legacy if legacy_masking else governance.mask
    strict_local = _strict_model(model)
    cost_routing = dict((routing or {}).get("costRouting") or {}) or None

    def classify_tool_output(value: str) -> list[dict[str, Any]]:
        return [
            row.wire()
            for row in governance.findings(
                {"tool_output": value},
                legacy=legacy_masking,
            )
        ]

    if routing:
        # First event, so the model badge updates before any token.
        yield chat_service.sse({"type": "privacy_route", **routing})
    if cost_routing:
        yield chat_service.sse({"type": "model_route", **cost_routing})
    if skills_event:
        yield chat_service.sse(skills_event)
    for step in context_steps or ():
        yield chat_service.sse(_step_event(step))
    try:
        async for event in _until_stopped(
            agent_service.run_turn(
                model["id"],
                messages,
                tools,
                ctx,
                tool_definitions=tool_definitions,
                temperature=temperature,
                sanitize_tool_output=(masker if sanitize_tool_output else None),
                sanitize_step_detail=masker if protect_enrichment else None,
                classify_tool_output=classify_tool_output if protect_enrichment else None,
                strict_local=strict_local,
                disable_fallbacks=disable_fallbacks,
                redact_logging=mask_at_rest,
                force_tool=force_tool,
            ),
            stopping,
        ):
            if event["type"] == "delta":
                text_parts.append(event["text"])
            elif event["type"] == "retract":
                # Narration the agent took back; see `agent.run_turn`.
                joined = "".join(text_parts).replace(event["text"], "", 1)
                text_parts[:] = [joined]
            elif event["type"] == "step":
                # Stored without `type`; one row per step id, the done event
                # replacing the running one.
                row = {k: v for k, v in event.items() if k != "type"}
                at = next((i for i, s in enumerate(steps) if s.get("id") == row.get("id")), None)
                if at is None:
                    steps.append(row)
                else:
                    steps[at] = row
            elif event["type"] == "usage":
                usage = {k: v for k, v in event.items() if k != "type"}
                continue  # re-emitted below with the credit figure
            elif event["type"] == "model_route":
                actual_model = str(event["actualModel"])
                routing = _with_actual_model(
                    routing or {},
                    str(event["routedModel"]),
                    actual_model,
                )
                yield chat_service.sse({"type": "privacy_route", **routing})
                if cost_routing:
                    cost_routing = {**cost_routing, "executedModel": actual_model}
                    routing = {**(routing or {}), "costRouting": cost_routing}
                    # Do not leak the agent loop's partial provider-route shape.
                    yield chat_service.sse({"type": "model_route", **cost_routing})
                continue
            elif event["type"] == "privacy_route" and event.get("source") == "tool_output":
                count = int(event.get("count") or 0)
                if event.get("action") == "mask_external":
                    tool_output_masked += count
                for finding in event.get("findings") or []:
                    key = (
                        str(finding.get("category") or "unknown"),
                        str(finding.get("source") or "tool_output"),
                    )
                    tool_output_findings[key] = tool_output_findings.get(key, 0) + int(
                        finding.get("count") or 0
                    )
                routing = {
                    **(routing or {}),
                    **(
                        {"initialAction": routing.get("action")}
                        if routing and "initialAction" not in routing
                        else {}
                    ),
                    "action": event.get("action") or "mask_external",
                    "toolOutputMasked": tool_output_masked,
                    "toolOutputFindings": [
                        {"category": category, "source": source, "count": total}
                        for (category, source), total in sorted(tool_output_findings.items())
                    ],
                }
                yield chat_service.sse({"type": "privacy_route", **routing})
                continue
            yield chat_service.sse(event)
        # A stopped turn still settles; only the label differs.
        if stopping.is_set():
            failed = "stopped"
    except chat_service.ChatStreamError as exc:
        log.warning("chat stream failed for session %s: %s", session_id, exc)
        failed = str(exc)
        yield chat_service.sse(
            _error_event(
                "모델 사용량 한도에 닿았습니다. 잠시 후 다시 시도해 주세요."
                if failed.startswith("upstream_429")
                else "모델 응답을 받지 못했습니다.",
                exc,
            )
        )
    except Exception as exc:  # noqa: BLE001 — turn still has to settle and close
        log.exception("chat stream crashed for session %s", session_id)
        failed = "internal_error"
        yield chat_service.sse(_error_event("요청 처리 중 오류가 발생했습니다.", exc))

    content = "".join(text_parts)
    if failed == "stopped" and not any(usage.values()):
        # A stopped stream never reaches the usage chunk; estimate, marked as one.
        usage = {
            "inputTokens": file_service.estimate_tokens(
                "".join(str(m.get("content") or "") for m in messages)
            ),
            "outputTokens": file_service.estimate_tokens(content),
            "estimated": True,
        }
    stored_content = masker(content)[0] if mask_at_rest or tool_output_findings else content
    protect_persistence = mask_at_rest or bool(tool_output_findings)
    # Tell the browser what was masked; it holds the streamed original.
    answer_findings = (
        governance.findings({"assistant_output": content}, legacy=legacy_masking)
        if stored_content != content
        else []
    )
    if answer_findings:
        routing = {
            **(routing or {}),
            "findingCounts": [
                *((routing or {}).get("findingCounts") or []),
                *(row.wire() for row in answer_findings),
            ],
        }
        yield chat_service.sse({"type": "privacy_route", **routing})
    stored_steps = _mask_text_tree(steps, masker) if protect_persistence else steps
    stored_actual_model = masker(actual_model)[0] if protect_persistence else actual_model
    credits = (
        0 if not content else charge_for_tokens(model, usage["inputTokens"], usage["outputTokens"])
    )
    if cost_routing:
        cost_routing = {**cost_routing, "executedModel": actual_model}
        if content and quality_model is not None and cost_routing.get("decision") == "routed":
            quality_credits = charge_for_tokens(
                quality_model,
                usage["inputTokens"],
                usage["outputTokens"],
            )
            cost_routing["estimatedCreditsSaved"] = max(0, quality_credits - credits)
        routing = {**(routing or {}), "costRouting": cost_routing}
    stored_routing = _mask_text_tree(routing, masker) if protect_persistence else routing

    # Stored and announced before the title call and the message transaction
    # below, both of which are free of this turn's artifact — so the panel
    # catches up close to when the closing text does, not well after it.
    new_artifact: str | None = None
    if stored_content and not failed:
        new_artifact = await _store_artifacts(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            content=stored_content,
            requested_artifacts=ctx.pending_artifacts,
            protect_privacy=protect_enrichment,
            legacy_masking=legacy_masking,
        )
        if new_artifact:
            yield chat_service.sse(
                {
                    "type": "artifact",
                    "artifactId": new_artifact,
                    # Whether the panel should open: only for an artifact that was asked for.
                    "deliberate": bool(ctx.pending_artifacts),
                }
            )

    title: str | None = None
    title_credits = 0
    title_model: str | None = None
    if is_first_turn and stored_content and not failed:
        enrichment = await _enrichment_model(
            model, strict_local=strict_local, disable_fallbacks=disable_fallbacks
        )
        title, title_usage = await chat_service.generate_title(
            enrichment["id"],
            first_user_message,
            stored_content,
            api_key,
            masker=masker if protect_enrichment else None,
            strict_local=strict_local,
            disable_fallbacks=disable_fallbacks,
            redact_logging=mask_at_rest or bool(tool_output_findings),
        )
        title_credits = charge_for_tokens(
            enrichment, title_usage["inputTokens"], title_usage["outputTokens"]
        )
        title_model = enrichment["id"]

    # One transaction: assistant message, deduction, title.
    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        user = await db.get(User, user_id)
        if session is not None and user is not None:
            if content:
                answer = Message(
                    session_id=session_id,
                    role=Role.assistant,
                    content=stored_content,
                    steps=stored_steps or None,
                    usage={**usage, "credits": credits},
                    model=stored_actual_model,
                    routing=stored_routing,
                    artifact_ids=[new_artifact] if new_artifact else None,
                    # A partial answer is kept and labelled by who ended it.
                    failure=(
                        TurnFailure.stopped
                        if failed == "stopped"
                        else TurnFailure.interrupted
                        if failed
                        else None
                    ),
                )
                db.add(answer)
                answer_id = answer.id
                settle(
                    db,
                    user,
                    credits,
                    reason="chat.completion",
                    session_id=session_id,
                    model=stored_actual_model,
                )
            else:
                # No answer: the question row carries the outcome and the retry.
                question = await db.get(Message, user_message_id) if user_message_id else None
                if question is not None:
                    # Stopped before the first token is still stopped, not unanswered.
                    question.failure = (
                        TurnFailure.stopped if failed == "stopped" else TurnFailure.no_answer
                    )
                    db.add(question)
            # Notes are kept from an empty completion but not from a failed turn.
            if ctx.pending_notes and not failed:
                await _store_notes(db, user_id, session_id, project_id, ctx.pending_notes)
            if title:
                session.title = title
            # Own ledger line: a different model may have run it.
            settle(
                db,
                user,
                title_credits,
                reason="chat.title",
                session_id=session_id,
                model=title_model,
            )
            if privacy_audit_id:
                privacy_audit = await db.get(AuditEvent, privacy_audit_id)
                if privacy_audit is not None:
                    privacy_audit.event_metadata = {
                        **(privacy_audit.event_metadata or {}),
                        **(stored_routing or {}),
                    }
                    db.add(privacy_audit)
            if routing_audit_id:
                routing_audit = await db.get(AuditEvent, routing_audit_id)
                if routing_audit is not None:
                    routing_audit.event_metadata = dict(cost_routing or {})
                    db.add(routing_audit)
            session.updated_at = utcnow()
            if new_artifact:
                session.artifact_id = new_artifact
            db.add(session)
            if tool_output_findings:
                db.add(
                    AuditEvent(
                        actor_id=user_id,
                        action=(
                            "privacy.mask_tool_output"
                            if tool_output_masked
                            else "privacy.strict_tool_output"
                        ),
                        target=session_id,
                        detail=governance.DETECTOR_VERSION,
                        event_metadata={
                            "detectorVersion": governance.DETECTOR_VERSION,
                            "policyVersion": governance.POLICY_VERSION,
                            "findings": [
                                {
                                    "category": category,
                                    "source": source,
                                    "count": count,
                                }
                                for (category, source), count in sorted(
                                    tool_output_findings.items()
                                )
                            ],
                            **(stored_routing or {}),
                        },
                        severity="warn",
                    )
                )
            await db.commit()

    # Own transaction, after the answer is durable. The artifact itself was
    # already stored and announced above; this is only the auto-memory pass,
    # which needs `answer_id` and is unrelated to what the panel shows.
    memory_step: dict | None = None
    if stored_content and not failed:
        memory_step = await _enrich_memory(
            user_id=user_id,
            session_id=session_id,
            content=stored_content,
            first_user_message=first_user_message,
            api_key=api_key,
            model=model,
            auto_memory=auto_memory,
            protect_privacy=protect_enrichment,
            strict_local=strict_local,
            disable_fallbacks=disable_fallbacks,
            redact_logging=mask_at_rest or bool(tool_output_findings),
            legacy_masking=legacy_masking,
            message_id=answer_id,
        )

    if memory_step:
        yield chat_service.sse(_step_event(memory_step))
    if cost_routing:
        yield chat_service.sse({"type": "model_route", **cost_routing})
    yield chat_service.sse({"type": "usage", **usage, "credits": credits})
    if title:
        yield chat_service.sse({"type": "title", "title": title})
    # The stored id travels with `done`; the browser's streaming id is provisional.
    yield chat_service.sse({"type": "done", **({"messageId": answer_id} if answer_id else {})})

    # Only this turn's signal; a newer turn may own the set.
    live = _STOPPING.get(session_id)
    if live is not None:
        live.discard(stopping)
        if not live:
            del _STOPPING[session_id]


@router.post("/{session_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_turn(session_id: str, user: CurrentUser, db: DbSession):
    """Asks the turn running on this session to stop.

    Idempotent. Closing the socket means the opposite: the answer is still wanted.
    """
    await _owned(db, user, session_id)
    # Every turn on the session; a superseded one may still be running.
    for signal in _STOPPING.get(session_id, set()):
        signal.set()


@router.post("/{session_id}/compare")
async def compare_models(
    session_id: str, payload: CompareRequest, request: Request, user: CurrentUser, db: DbSession
):
    """Runs one prompt against two or three models and streams all of them.

    Every column is a real completion on the caller's key, billed separately
    and stored on one assistant message.
    """
    session = await _owned(db, user, session_id)
    if session.kind is not SessionKind.chat:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="surface_not_implemented"
        )

    # Policy first; no cached default may authorise a comparison.
    policy = await _require_egress_policy()
    content = payload.content
    if policy.intent_filter and (hit := governance.blocked_by(content, policy.blocked_categories)):
        await _audit_policy(user, request, "filter.blocked", hit)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"blocked_category:{hit}"
        )

    catalogue = await model_service.list_models_for_egress()
    if len(set(payload.models)) != len(payload.models):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="models_must_be_distinct"
        )
    catalogue_models = catalogue["models"]
    chosen = [model_service.find(catalogue_models, model_id) for model_id in payload.models]
    if any(model is None or "chat" not in model.get("kinds", []) for model in chosen):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="models_unavailable")
    allowed_ids = set(user.allowed_models or [])
    if allowed_ids and any(model_id not in allowed_ids for model_id in payload.models):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="model_not_allowed",
        )
    # Request order is part of the decision-token binding.
    chosen = [model for model in chosen if model is not None]

    history = await _history(db, session.id)
    attachment_rows, attachment_meta = await _owned_attachments(db, user, payload.attachments)
    try:
        workspace = await assemble(
            db,
            user,
            session,
            attachment_ids=payload.attachments,
            # The smallest window among the compared models decides how much file goes in.
            file_budget=min(file_budget(m) for m in chosen),
            activated_skill_ids=payload.activated_skill_ids,
            starting_template_id=payload.starting_template_id,
            question=content,
            # Comparison exposes no tools.
            available_tool_names=set(),
        )
    except WorkspaceContextError as exc:
        _raise_workspace_error(exc)
    resolved = await _resolve_privacy(
        user=user,
        session=session,
        policy=policy,
        catalogue=catalogue_models,
        requested=chosen,
        sources=_privacy_sources(content, history, workspace.blocks),
        explicit_action=payload.privacy_action,
        decision_token=payload.privacy_decision_token,
    )
    if isinstance(resolved, JSONResponse):
        await _audit_policy(
            user,
            request,
            "privacy.decision_required",
            governance.DETECTOR_VERSION,
            metadata={
                **_decision_audit_metadata(resolved, session.id),
                "compare": True,
            },
        )
        return resolved
    chosen = resolved.models

    # Headroom checked only after a possible collapse to strict-local.
    if not has_headroom(user, max(chosen, key=lambda m: m["creditCost"])):
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="no_credits")

    masker = governance.mask_legacy if policy.pii_masking else governance.mask
    stored_content = masker(content)[0] if resolved.findings else content
    if resolved.findings and attachment_meta:
        attachment_meta = [
            {
                **item,
                "name": masker(str(item.get("name") or ""))[0],
                "error": (
                    masker(str(item["error"]))[0] if item.get("error") else item.get("error")
                ),
            }
            for item in attachment_meta
        ]
    outbound_history = [message.content for message in history]
    trusted_context = workspace.trusted
    untrusted_context = workspace.untrusted
    if resolved.mask_outbound:
        content = masker(content)[0]
        outbound_history = _mask_list(outbound_history, legacy=policy.pii_masking)
        trusted_context = _mask_list(trusted_context, legacy=policy.pii_masking)
        untrusted_context = _mask_list(untrusted_context, legacy=policy.pii_masking)

    for stored in attachment_rows:
        stored.session_id = session.id
        db.add(stored)

    db.add(
        Message(
            session_id=session.id,
            role=Role.user,
            content=stored_content,
            attachments=attachment_meta,
            routing=resolved.routing,
            started_from=workspace.started_from,
        )
    )
    session.updated_at = utcnow()
    if not session.title:
        session.title = chat_service.provisional_title(stored_content)
    db.add(session)
    privacy_audit_id: str | None = None
    if resolved.findings:
        privacy_audit = AuditEvent(
            actor_id=user.id,
            action=f"privacy.{resolved.action}",
            target=session.id,
            detail=governance.DETECTOR_VERSION,
            event_metadata={
                **governance.finding_metadata(resolved.findings),
                **resolved.routing,
            },
            ip=client_ip(request),
            severity="warn",
        )
        privacy_audit_id = privacy_audit.id
        db.add(privacy_audit)
    await db.commit()

    await litellm_service.ensure_key(user)
    if db.is_modified(user):
        db.add(user)
        await db.commit()
    _, api_key = await litellm_service.credentials_for(user)

    wire = [
        {"role": message.role.value, "content": body}
        for message, body in zip(history, outbound_history, strict=True)
    ]
    wire.append({"role": "user", "content": content})
    messages = build_messages(
        session.kind,
        wire,
        with_tools=False,
        web_search=False,
        extra=trusted_context,
        untrusted_context=untrusted_context,
    )

    return StreamingResponse(
        _heartbeat(
            _run_comparison(
                user_id=user.id,
                api_key=api_key,
                session_id=session.id,
                models=chosen,
                messages=messages,
                skills_event=workspace.skills_event(),
                context_steps=_context_steps(workspace),
                routing=resolved.routing,
                mask_at_rest=policy.pii_masking or policy.external_data_guard,
                legacy_masking=policy.pii_masking,
                privacy_audit_id=privacy_audit_id,
            ),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _run_comparison(
    *,
    user_id: str,
    api_key: str,
    session_id: str,
    models: list[dict],
    messages: list[dict],
    skills_event: dict | None = None,
    context_steps: list[dict] | None = None,
    routing: dict,
    mask_at_rest: bool = False,
    legacy_masking: bool = False,
    privacy_audit_id: str | None = None,
) -> AsyncIterator[str]:
    """Fans out, merges the streams, then settles every column in one transaction.

    Interleaved on one connection: the turn is stored and billed as one thing
    even when a column fails.
    """
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    routing_holder = {"value": routing}
    results: dict[str, dict] = {
        model["id"]: {
            "model": model["id"],
            "routedModel": model["id"],
            "actualModel": None,
            "dataBoundary": model.get("dataBoundary") or "unknown",
            "content": "",
            "usage": None,
            "error": None,
        }
        for model in models
    }

    yield chat_service.sse({"type": "privacy_route", **routing})

    async def run(model: dict) -> None:
        slot = results[model["id"]]
        try:
            async for event in chat_service.stream_completion(
                model["id"],
                messages,
                user_id,
                api_key,
                strict_local=_strict_model(model),
                redact_logging=mask_at_rest,
            ):
                if event["type"] == "delta":
                    slot["content"] += event["text"]
                    await queue.put(
                        {"type": "variant", "model": model["id"], "text": event["text"]}
                    )
                elif event["type"] == "retract":
                    slot["content"] = slot["content"].replace(event["text"], "", 1)
                    await queue.put(
                        {"type": "variant_retract", "model": model["id"], "text": event["text"]}
                    )
                elif event["type"] == "usage":
                    slot["usage"] = {k: v for k, v in event.items() if k != "type"}
                elif event["type"] == "model_route":
                    slot["actualModel"] = str(event["actualModel"])
                    routing_holder["value"] = _with_actual_model(
                        routing_holder["value"],
                        model["id"],
                        slot["actualModel"],
                    )
                    await queue.put({"type": "privacy_route", **routing_holder["value"]})
        except Exception as exc:  # noqa: BLE001 — one dead column must not kill the row
            log.warning("comparison column failed (%s): %s", model["id"], exc)
            slot["error"] = "모델 응답을 받지 못했습니다."
        finally:
            usage = slot["usage"] or {"inputTokens": 0, "outputTokens": 0}
            credits = (
                0
                if not slot["content"]
                else charge_for_tokens(model, usage["inputTokens"], usage["outputTokens"])
            )
            slot["credits"] = credits
            await queue.put(
                {
                    "type": "variant_done",
                    "model": model["id"],
                    "routedModel": model["id"],
                    "actualModel": slot["actualModel"] or model["id"],
                    "credits": credits,
                    "error": slot["error"],
                    **usage,
                }
            )

    async def drive() -> None:
        await asyncio.gather(*(run(m) for m in models))
        await queue.put(None)

    task = asyncio.create_task(drive())
    try:
        if skills_event:
            yield chat_service.sse(skills_event)
        for step in context_steps or ():
            yield chat_service.sse(_step_event(step))
        while (event := await queue.get()) is not None:
            yield chat_service.sse(event)
    finally:
        await task

    masker = governance.mask_legacy if legacy_masking else governance.mask
    variants = [
        {
            "model": r["model"],
            "routedModel": r["routedModel"],
            "actualModel": r["actualModel"] or r["routedModel"],
            "dataBoundary": r["dataBoundary"],
            "content": r["content"],
            "credits": r.get("credits", 0),
            "usage": r["usage"],
            "error": r["error"],
        }
        for r in results.values()
    ]
    if mask_at_rest:
        variants = _mask_text_tree(variants, masker)
    stored_routing = (
        _mask_text_tree(routing_holder["value"], masker)
        if mask_at_rest
        else routing_holder["value"]
    )
    total = sum(v["credits"] for v in variants)

    # First successful column is the default answer; `choose_variant` can change it.
    chosen = next((v for v in variants if v["content"] and not v["error"]), None)
    for variant in variants:
        variant["chosen"] = variant is chosen
    stored_prelude = _prelude_steps(skills_event, context_steps)
    if stored_prelude and mask_at_rest:
        stored_prelude = _mask_text_tree(stored_prelude, masker)

    answer = Message(
        session_id=session_id,
        role=Role.assistant,
        content=chosen["content"] if chosen else "",
        variants=variants,
        usage={"credits": total},
        steps=stored_prelude or None,
        model=chosen["actualModel"] if chosen else None,
        routing=stored_routing,
    )
    async with SessionLocal() as db:
        db.add(answer)
        settled = await db.get(User, user_id)
        if privacy_audit_id:
            privacy_audit = await db.get(AuditEvent, privacy_audit_id)
            if privacy_audit is not None:
                privacy_audit.event_metadata = {
                    **(privacy_audit.event_metadata or {}),
                    **stored_routing,
                }
                db.add(privacy_audit)
        if settled is not None and total:
            # No model: one charge covers several.
            settle(db, settled, total, reason="chat.compare", session_id=session_id)
        await db.commit()

    # The stored id; the browser's streaming id is provisional.
    yield chat_service.sse({"type": "done", "credits": total, "messageId": answer.id})


@router.post("/{session_id}/messages/{message_id}/variant", response_model=MessageOut)
async def choose_variant(
    session_id: str, message_id: str, payload: ChooseVariant, user: CurrentUser, db: DbSession
):
    """Marks which comparison answer the conversation continues from."""
    await _owned(db, user, session_id)
    message = await db.get(Message, message_id)
    if message is None or message.session_id != session_id or not message.variants:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="variant_not_found")

    variants = [dict(v) for v in message.variants]
    if not any(v.get("model") == payload.model for v in variants):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unknown_model")

    for variant in variants:
        variant["chosen"] = variant.get("model") == payload.model
    message.variants = variants
    message.content = next(v.get("content") or "" for v in variants if v["chosen"])
    selected = next(v for v in variants if v["chosen"])
    message.model = selected.get("actualModel") or payload.model
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return MessageOut.of(message)


async def _store_artifacts(
    *,
    user_id: str,
    session_id: str,
    project_id: str,
    content: str,
    requested_artifacts: list[dict] | None = None,
    protect_privacy: bool = False,
    legacy_masking: bool = False,
) -> str | None:
    """Artifacts derived from a finished turn; never raises.

    Its own transaction, committed the moment it is done, and called before
    the turn's message is even persisted — not after auto-memory, which has
    nothing to do with what this panel shows. The version this replaced ran
    both in one pass, so the panel only caught up well after the closing
    text was already sitting on screen looking done.
    """
    privacy_masker = (
        (governance.mask_legacy if legacy_masking else governance.mask) if protect_privacy else None
    )
    artifact_id: str | None = None
    async with SessionLocal() as db:
        try:
            # A `create_artifact` call wins over extraction from the transcript.
            # Both run: a turn can do each once.
            requested_id = await artifact_extract.store_requested(
                db,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                requests=requested_artifacts or [],
                masker=privacy_masker,
            )
            extracted_id = await artifact_extract.extract(
                db,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                content=privacy_masker(content)[0] if privacy_masker else content,
            )
            artifact_id = requested_id or extracted_id
            await db.commit()
        except Exception:  # noqa: BLE001
            log.exception("artifact extraction failed for session %s", session_id)
            return None
    return artifact_id


async def _enrich_memory(
    *,
    user_id: str,
    session_id: str,
    content: str,
    first_user_message: str,
    api_key: str,
    model: dict,
    auto_memory: bool,
    protect_privacy: bool = False,
    strict_local: bool = False,
    disable_fallbacks: bool = False,
    redact_logging: bool = False,
    legacy_masking: bool = False,
    message_id: str | None = None,
) -> dict | None:
    """Facts auto-memory pulls from a finished turn; never raises.

    Own transaction, after the answer is durable — the memory step is
    appended to the stored message. Unrelated to the artifact, which is
    stored separately by `_store_artifacts` before this even starts.
    """
    if not auto_memory:
        return None
    memory_step: dict | None = None
    async with SessionLocal() as db:
        user = await db.get(User, user_id)
        if user is None:
            return None
        privacy_masker = (
            (governance.mask_legacy if legacy_masking else governance.mask)
            if protect_privacy
            else None
        )
        try:
            enrichment = await _enrichment_model(
                model, strict_local=strict_local, disable_fallbacks=disable_fallbacks
            )
            written, spent = await auto_memory_service.extract(
                db,
                user,
                user_message=first_user_message,
                assistant_message=content,
                api_key=api_key,
                model=enrichment["id"],
                masker=privacy_masker,
                strict_local=strict_local,
                disable_fallbacks=disable_fallbacks,
                redact_logging=redact_logging,
            )
            if written:
                log.info("auto-memory wrote %d fact(s) for user %s", written, user.id)
                memory_step = _memory_saved_step(written)
                message = await db.get(Message, message_id) if message_id else None
                if message is not None:
                    message.steps = [*(message.steps or []), memory_step]
                    db.add(message)
            # Charged whether or not a fact came out.
            settle(
                db,
                user,
                charge_for_tokens(enrichment, spent["inputTokens"], spent["outputTokens"]),
                reason="chat.memory",
                session_id=session_id,
                model=enrichment["id"],
            )
        except Exception:  # noqa: BLE001
            log.exception("auto-memory failed for session %s", session_id)
            return None

        try:
            await db.commit()
        except Exception:  # noqa: BLE001
            log.exception("memory commit failed for session %s", session_id)
            return None
    return memory_step


async def _audit_policy(
    user: User,
    request: Request,
    action: str,
    detail: str,
    *,
    metadata: dict | None = None,
) -> None:
    """Writes a policy event on its own connection.

    Separate from the turn's transaction: the record of a refusal must survive
    the request.
    """
    audit_target = (
        str(metadata["sessionId"]) if metadata and metadata.get("sessionId") else user.email
    )
    async with SessionLocal() as db:
        db.add(
            AuditEvent(
                actor_id=user.id,
                action=action,
                target=audit_target,
                detail=detail,
                event_metadata=metadata,
                ip=client_ip(request),
                severity="warn",
            )
        )
        await db.commit()


async def _run_page(
    *,
    outline_model: dict | None = None,
    #: `None` plans and offers; anything else is written as approved.
    approved_plan: dict | None = None,
    #: False after "있는 자료로 진행", so the writer does not ask again.
    may_ask: bool = True,
    #: Carried so a stored proposal is later written against the same files.
    attachments: list[str] | None = None,
    #: Carried with the proposal for the same reason.
    answers: dict[str, str] | None = None,
    user_id: str,
    api_key: str,
    session_id: str,
    model: dict,
    request: str,
    project_id: str | None,
    routing: dict[str, Any] | None = None,
    template: design_templates.DesignTemplate,
    trusted_context: list[str] | None = None,
    untrusted_context: list[str] | None = None,
    design_tokens: dict[str, str] | None = None,
    skills_event: dict | None = None,
    context_steps: list[dict] | None = None,
    #: Composer's search toggle; off on a strict-local route.
    web_search: bool = True,
    project_sources: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Drives one HTML artifact to completion and settles it.

    A half-written page is still stored.
    """
    blocks: list[dict] = []
    proposal: dict | None = None
    questions: list[dict] | None = None
    html = ""
    usage = {"inputTokens": 0, "outputTokens": 0}
    doc_title = ""
    #: What the research pass read; citations belong to the document.
    sources: list[dict] = []
    research_log: dict[str, Any] | None = None

    if routing:
        # First, so the model badge updates.
        yield chat_service.sse({"type": "privacy_route", **routing})
    if skills_event:
        yield chat_service.sse(skills_event)
    for step in context_steps or ():
        yield chat_service.sse(_step_event(step))
    try:
        stream = page_service.write(
            web_search=web_search,
            request=request,
            approved_plan=approved_plan,
            may_ask=may_ask,
            model=model["id"],
            outline_model=(outline_model or {}).get("id", ""),
            api_key=api_key,
            template=template,
            tokens=design_tokens,
            trusted_context=trusted_context,
            untrusted_context=untrusted_context,
            project_sources=project_sources,
        )
        async for event in stream:
            if event["type"] in ("proposal", "needs"):
                # Stopped on purpose: forwarded, and stored below instead of an artifact.
                if event["type"] == "proposal":
                    proposal = event["plan"]
                else:
                    questions = event["questions"]
                yield chat_service.sse(event)
                continue
            if event["type"] == "page":
                html = event["html"]
                blocks = event["blocks"]
                continue
            if event["type"] == "sources":
                sources = list(event.get("sources") or [])
            if event["type"] == "research":
                research_log = dict(event.get("research") or {})
            if event["type"] == "title":
                doc_title = str(event.get("title") or "").strip()
            if event["type"] == "usage":
                usage = {k: v for k, v in event.items() if k != "type"}
                continue
            yield chat_service.sse(event)
    except Exception as exc:  # noqa: BLE001 — the turn must still settle
        log.exception("page generation crashed for session %s", session_id)
        yield chat_service.sse(_error_event("문서를 만들지 못했습니다.", exc))

    # Planned or asked: stored on the session, no artifact written.
    if proposal is not None or questions is not None:
        await _settle_plan_turn(
            session_id=session_id,
            user_id=user_id,
            request=request,
            attachments=list(attachments or []),
            answers=dict(answers or {}),
            model=model,
            outline_model=outline_model,
            usage=usage,
            proposal=proposal,
            questions=questions,
        )
        yield chat_service.sse({"type": "usage", **usage, "credits": 0})
        yield chat_service.sse({"type": "done"})
        return

    written = page_service.filled(blocks)
    credits = (
        0
        if not written
        else charge_for_tokens(model, usage["inputTokens"], usage["outputTokens"])
        # The planner's own tokens at the planner's own price, when one ran.
        + charge_for_tokens(
            outline_model or model,
            usage.get("outlineInputTokens", 0),
            usage.get("outlineOutputTokens", 0),
        )
    )

    artifact_id: str | None = None
    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        user = await db.get(User, user_id)
        if session is not None and user is not None:
            title = (doc_title or session.title or request.strip()[:60] or template.name)[:200]
            if written and html and session.kind is SessionKind.report:
                # Report blocks are stored as HTML sections so the page view can
                # re-render them under any template; decks keep the html artifact.
                artifact_id = await _store_document(
                    db,
                    session,
                    user_id=user_id,
                    project_id=project_id,
                    kind=ArtifactKind.report,
                    title=title,
                    summary=_regeneration_summary(request),
                    data={
                        "kind": "report",
                        "templateId": template.id,
                        "sections": [
                            {
                                "id": f"s{index}_{uuid4().hex[:6]}",
                                "heading": block["title"],
                                "level": 1,
                                "status": "done",
                                "content": block["html"],
                                "format": "html",
                            }
                            for index, block in enumerate(blocks)
                            if block.get("layout") != "cover"
                        ],
                        "sources": sources,
                        **({"research": research_log} if research_log is not None else {}),
                        "citationStyle": "APA",
                        "lint": lint.wire(
                            lint.check(
                                lint.from_blocks(blocks),
                                slides=False,
                                limits=template.limits,
                            )
                        ),
                        "wordCount": sum(
                            len(re.sub(r"<[^>]+>", " ", b.get("html") or "").split())
                            for b in blocks
                        ),
                        **({"design": design_tokens} if design_tokens else {}),
                    },
                )
                session.artifact_id = artifact_id
                session.pending = None
            elif written and html:
                artifact_id = await _store_document(
                    db,
                    session,
                    user_id=user_id,
                    project_id=project_id,
                    kind=ArtifactKind.html,
                    title=title,
                    summary=_regeneration_summary(request),
                    data={
                        "kind": "html",
                        "language": "html",
                        "content": html,
                        "templateId": template.id,
                        # Blocks are the source, `content` the render; both kept
                        # for per-block rewrites.
                        "blocks": [
                            {"title": b["title"], "layout": b["layout"], "html": b["html"]}
                            for b in blocks
                        ],
                        "lint": lint.wire(
                            lint.check(
                                lint.from_blocks(blocks),
                                slides=template.kind == "deck",
                                limits=template.limits,
                            )
                        ),
                        **({"design": design_tokens} if design_tokens else {}),
                    },
                )
                session.artifact_id = artifact_id
                # The proposal has become the document.
                session.pending = None

                db.add(
                    Message(
                        session_id=session_id,
                        role=Role.assistant,
                        content=f"{template.name}으로 {len(written)}개 부분을 작성했습니다.",
                        usage={**usage, "credits": credits},
                        model=model["id"],
                        steps=_prelude_steps(skills_event, context_steps) or None,
                        routing=routing,
                    )
                )
                settle(
                    db,
                    user,
                    credits,
                    reason="page.generate",
                    session_id=session_id,
                    model=model["id"],
                )
            session.updated_at = utcnow()
            db.add(session)
            await db.commit()

    if artifact_id:
        yield chat_service.sse({"type": "artifact", "artifactId": artifact_id})
    yield chat_service.sse({"type": "usage", **usage, "credits": credits})
    yield chat_service.sse({"type": "done"})


async def _run_deck(
    *,
    #: Deck template: rules to the planner, `look` as the face, id on the artifact for export.
    template: design_templates.DesignTemplate | None = None,
    outline_model: dict | None = None,
    #: `None` plans and offers; anything else is written as approved.
    approved_plan: dict | None = None,
    #: False after "있는 자료로 진행", so the writer does not ask again.
    may_ask: bool = True,
    #: Carried so a stored proposal is later written against the same files.
    attachments: list[str] | None = None,
    #: Carried with the proposal for the same reason.
    answers: dict[str, str] | None = None,
    user_id: str,
    api_key: str,
    session_id: str,
    model: dict,
    request: str,
    project_id: str | None,
    routing: dict[str, Any] | None = None,
    trusted_context: list[str] | None = None,
    untrusted_context: list[str] | None = None,
    design_tokens: dict[str, str] | None = None,
    skills_event: dict | None = None,
    context_steps: list[dict] | None = None,
    #: Composer's search toggle; off on a strict-local route.
    web_search: bool = True,
    #: Pictures agreed to on the figure card: `None` on the planning pass, `[]`
    #: when declined. Only the report draws them.
    figures_plan: list[dict] | None = None,
    #: Draws the figures; the image default, not the writer.
    image_model: dict | None = None,
) -> AsyncIterator[str]:
    """Drives one deck to completion and settles it.

    Same contract as `_run_report`: the deck is an artifact, not a chat message.
    """
    slides: list[dict] = []
    proposal: dict | None = None
    questions: list[dict] | None = None
    usage = {"inputTokens": 0, "outputTokens": 0}
    doc_title = ""

    if routing:
        # First, so the model badge updates.
        yield chat_service.sse({"type": "privacy_route", **routing})
    if skills_event:
        yield chat_service.sse(skills_event)
    for step in context_steps or ():
        yield chat_service.sse(_step_event(step))
    if template is not None and template.instructions.strip():
        # Template genre rules reach the planner; its typesetting rules do not.
        trusted_context = [f"[서식: {template.name}]\n{template.instructions.strip()}"] + list(
            trusted_context or []
        )
    try:
        stream = deck_service.write(
            web_search=web_search,
            figures_plan=figures_plan,
            image_model=image_model,
            request=request,
            approved_plan=approved_plan,
            may_ask=may_ask,
            model=model["id"],
            outline_model=(outline_model or {}).get("id", ""),
            api_key=api_key,
            trusted_context=trusted_context,
            untrusted_context=untrusted_context,
            tokens=design_tokens,
        )
        async for event in stream:
            if event["type"] in ("proposal", "needs"):
                # Stopped on purpose: forwarded, and stored below instead of an artifact.
                if event["type"] == "proposal":
                    proposal = event["plan"]
                else:
                    questions = event["questions"]
                yield chat_service.sse(event)
                continue
            if event["type"] == "deck":
                slides = event["slides"]
                continue
            if event["type"] == "title":
                doc_title = str(event.get("title") or "").strip()
            if event["type"] == "usage":
                usage = {k: v for k, v in event.items() if k != "type"}
                continue
            yield chat_service.sse(event)
    except Exception as exc:  # noqa: BLE001 — the turn must still settle
        log.exception("deck generation crashed for session %s", session_id)
        yield chat_service.sse(_error_event("슬라이드를 만들지 못했습니다.", exc))

    # Planned or asked: stored on the session, no artifact written.
    if proposal is not None or questions is not None:
        drawn = (proposal or {}).pop("figures", None)
        await _settle_plan_turn(
            session_id=session_id,
            user_id=user_id,
            request=request,
            attachments=list(attachments or []),
            answers=dict(answers or {}),
            model=model,
            outline_model=outline_model,
            usage=usage,
            proposal=proposal,
            questions=questions,
            figures={"plan": proposal, **drawn} if drawn and proposal else None,
        )
        yield chat_service.sse({"type": "usage", **usage, "credits": 0})
        yield chat_service.sse({"type": "done"})
        return

    written = deck_service.filled(slides)
    credits = (
        0
        if not written
        else charge_for_tokens(model, usage["inputTokens"], usage["outputTokens"])
        # The planner's own tokens at the planner's own price, when one ran.
        + charge_for_tokens(
            outline_model or model,
            usage.get("outlineInputTokens", 0),
            usage.get("outlineOutputTokens", 0),
        )
    )

    artifact_id: str | None = None
    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        user = await db.get(User, user_id)
        if session is not None and user is not None:
            title = (doc_title or session.title or request.strip()[:60] or "슬라이드")[:200]
            if written:
                artifact_design = design_tokens
                if template is not None and template.look:
                    # The template's look outranks the project default and the request.
                    artifact_design = design_service.normalise_tokens(
                        {**(design_tokens or {}), "visualStyle": template.look}
                    )
                elif not artifact_design:
                    requested_style = str(
                        (approved_plan or {}).get("visualStyle")
                        or design_service.visual_style_for(request)
                    )
                    if requested_style != "editorial":
                        artifact_design = design_service.normalise_tokens(
                            {"visualStyle": requested_style}
                        )
                artifact_id = await _store_document(
                    db,
                    session,
                    user_id=user_id,
                    project_id=project_id,
                    kind=ArtifactKind.deck,
                    title=title,
                    summary=_regeneration_summary(request),
                    data={
                        "kind": "deck",
                        "theme": "기본",
                        # Export uses the template's PowerPoint half.
                        **({"templateId": template.id} if template else {}),
                        # Snapshot: a later project change must not repaint the deck.
                        **({"design": artifact_design} if artifact_design else {}),
                        "lint": lint.wire(lint.check(lint.from_slides(slides), slides=True)),
                        # Every slide, including unwritten ones — a gap stays
                        # visible so it can be fixed.
                        "slides": slides,
                    },
                )
                session.artifact_id = artifact_id
                # The proposal has become the document.
                session.pending = None

                db.add(
                    Message(
                        session_id=session_id,
                        role=Role.assistant,
                        content=f"{len(written)}장짜리 슬라이드를 만들었습니다.",
                        usage={**usage, "credits": credits},
                        model=model["id"],
                        steps=_prelude_steps(skills_event, context_steps) or None,
                        routing=routing,
                    )
                )
                settle(
                    db,
                    user,
                    credits,
                    reason="deck.generate",
                    session_id=session_id,
                    model=model["id"],
                )
            else:
                # Nothing written: record the failure so a reload shows it.
                db.add(
                    Message(
                        session_id=session_id,
                        role=Role.assistant,
                        content="",
                        failure=TurnFailure.no_answer,
                        model=model["id"],
                        steps=_prelude_steps(skills_event, context_steps) or None,
                        routing=routing,
                    )
                )
            session.updated_at = utcnow()
            db.add(session)
            await db.commit()

    if artifact_id:
        yield chat_service.sse({"type": "artifact", "artifactId": artifact_id})
    yield chat_service.sse({"type": "usage", **usage, "credits": credits})
    yield chat_service.sse({"type": "done"})


async def _revise_document(
    *,
    user_id: str,
    api_key: str,
    session_id: str,
    model: dict,
    instruction: str,
    routing: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Applies one instruction to the document on screen.

    `services.revise` says which parts it lands on; those are rewritten with the
    surface's own machinery. Failures are reported, never turned into a regeneration.
    """
    usage = {"inputTokens": 0, "outputTokens": 0}
    if routing:
        yield chat_service.sse({"type": "privacy_route", **routing})

    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        artifact = await db.get(Artifact, session.artifact_id) if session else None
        if session is None or artifact is None or not artifact.data:
            yield chat_service.sse({"type": "error", "message": "고칠 문서를 찾지 못했습니다."})
            yield chat_service.sse({"type": "done"})
            return
        kind = artifact.kind
        data = dict(artifact.data)
        title = artifact.title or ""
        # The request the document was written from and the files it drew on. A rewrite
        # given only the title reaches for the pen where the original read the material,
        # and a table of measured values comes back as a table of plausible ones.
        turns = (
            await db.exec(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(col(Message.created_at))
            )
        ).all()
        asked = [m for m in turns if m.role is Role.user and (m.content or "").strip()]
        request = str(asked[0].content if asked else session.title or "")
        file_ids = list(
            dict.fromkeys(
                str(a.get("id"))
                for m in asked
                for a in (m.attachments or [])
                if isinstance(a, dict) and a.get("id")
            )
        )
        material: list[str] = []
        if file_ids:
            files = (
                await db.exec(
                    select(StoredFile).where(
                        col(StoredFile.id).in_(file_ids), StoredFile.user_id == user_id
                    )
                )
            ).all()
            budget = file_budget(model)
            for stored in sorted(files, key=lambda f: file_ids.index(f.id)):
                if not stored.text or budget <= 0:
                    continue
                text = stored.text[:budget]
                budget -= len(text)
                material.append(f"## {stored.name}\n{text}")

    is_deck = kind is ArtifactKind.deck
    parts = list(data.get("slides") if is_deck else data.get("sections") or [])
    names = [str(p.get("title" if is_deck else "heading") or "") for p in parts]
    if not parts:
        yield chat_service.sse({"type": "error", "message": "고칠 내용이 없습니다."})
        yield chat_service.sse({"type": "done"})
        return

    yield chat_service.sse(
        {"type": "step", "id": "route", "label": "무엇을 고칠지 보는 중", "status": "running"}
    )
    plan = await revise.plan(
        message=instruction, title=title, parts=names, model=model["id"], api_key=api_key
    )
    usage["inputTokens"] += plan.usage["inputTokens"]
    usage["outputTokens"] += plan.usage["outputTokens"]
    if not plan.revises:
        # A new-document request is said, not acted on.
        yield chat_service.sse(
            {"type": "step", "id": "route", "label": "새 문서 요청", "status": "done"}
        )
        yield chat_service.sse(
            {
                "type": "delta",
                "text": "이건 지금 문서를 고치는 요청으로 보이지 않습니다. "
                "새로 쓰려면 '새로 써 줘' 라고 알려 주세요.",
            }
        )
        yield chat_service.sse({"type": "usage", **usage, "credits": 0})
        yield chat_service.sse({"type": "done"})
        return

    yield chat_service.sse(
        {
            "type": "step",
            "id": "route",
            "label": revise.label(plan, names),
            "status": "done",
        }
    )

    changed = 0
    for index in plan.targets:
        part = parts[index]
        label = names[index] or f"{index + 1}"
        yield chat_service.sse(
            {"type": "step", "id": f"r{index}", "label": label, "status": "running"}
        )
        try:
            if is_deck:
                written, spent = await deck_service.rewrite_slide(
                    request=request,
                    slides=parts,
                    target_id=str(part.get("id") or ""),
                    model=model["id"],
                    api_key=api_key,
                    note=plan.note,
                    material=material,
                    typed=instruction,
                )
                parts[index] = written
                yield chat_service.sse({"type": "slide", "slide": written, "done": True})
            else:
                body, spent = await report_service.rewrite_section(
                    request=request,
                    heading=label,
                    sections=richtext.normalise(parts),
                    target_id=str(part.get("id") or ""),
                    model=model["id"],
                    api_key=api_key,
                    note=plan.note,
                    sources=list(data.get("sources") or []),
                    material=material,
                )
                if not body.strip():
                    raise ValueError("빈 결과")
                # The rewrite is Markdown; an old `html` flag would show literal asterisks.
                parts[index] = {
                    **part,
                    "content": richtext.tidy_tables(body),
                    "format": "markdown",
                    "status": "done",
                }
                parts[index].pop("factCheck", None)
                yield chat_service.sse(
                    {
                        "type": "section",
                        "sectionId": part.get("id"),
                        "heading": label,
                        "content": body,
                        "done": True,
                    }
                )
        except Exception as exc:  # noqa: BLE001 — one bad part is not a failed turn
            log.warning("revision of %r failed: %s", label, exc)
            yield chat_service.sse(
                {"type": "step", "id": f"r{index}", "label": label, "status": "error"}
            )
            continue
        usage["inputTokens"] += spent["inputTokens"]
        usage["outputTokens"] += spent["outputTokens"]
        changed += 1
        yield chat_service.sse(
            {"type": "step", "id": f"r{index}", "label": label, "status": "done"}
        )

    if not changed:
        yield chat_service.sse({"type": "error", "message": "고치지 못했습니다."})
        yield chat_service.sse({"type": "usage", **usage, "credits": 0})
        yield chat_service.sse({"type": "done"})
        return

    credits = charge_for_tokens(model, usage["inputTokens"], usage["outputTokens"])
    artifact_id: str | None = None
    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        user = await db.get(User, user_id)
        if session is not None and user is not None:
            data["slides" if is_deck else "sections"] = parts
            if not is_deck:
                data["wordCount"] = report_service.word_count(parts)
                data["lint"] = lint.wire(lint.check(lint.from_sections(parts)))
            # `_store_document` snapshots the previous body as a version.
            artifact_id = await _store_document(
                db,
                session,
                user_id=user_id,
                project_id=session.project_id,
                kind=kind,
                title=title or "문서",
                summary=instruction.strip()[:80] or "문서 수정",
                data=data,
            )
            session.artifact_id = artifact_id
            db.add(
                Message(
                    session_id=session_id,
                    role=Role.assistant,
                    content=f"{changed}곳을 고쳤습니다.",
                    usage={**usage, "credits": credits},
                    model=model["id"],
                    routing=routing,
                )
            )
            settle(
                db,
                user,
                credits,
                reason="document.revise",
                session_id=session_id,
                model=model["id"],
                surface=session.kind.value,
            )
            session.updated_at = utcnow()
            db.add(session)
            await db.commit()

    if artifact_id:
        yield chat_service.sse({"type": "artifact", "artifactId": artifact_id})
    yield chat_service.sse({"type": "usage", **usage, "credits": credits})
    yield chat_service.sse({"type": "done"})


async def _run_report(
    *,
    template: design_templates.DesignTemplate | None = None,
    outline_model: dict | None = None,
    #: `None` plans and offers; anything else is written as approved.
    approved_plan: dict | None = None,
    #: False after "있는 자료로 진행", so the writer does not ask again.
    may_ask: bool = True,
    #: Carried so a stored proposal is later written against the same files.
    attachments: list[str] | None = None,
    #: Carried with the proposal for the same reason.
    answers: dict[str, str] | None = None,
    user_id: str,
    api_key: str,
    session_id: str,
    model: dict,
    request: str,
    project_id: str | None,
    routing: dict[str, Any] | None = None,
    trusted_context: list[str] | None = None,
    untrusted_context: list[str] | None = None,
    design_tokens: dict[str, str] | None = None,
    skills_event: dict | None = None,
    context_steps: list[dict] | None = None,
    #: Composer's search toggle; off on a strict-local route.
    web_search: bool = True,
    #: Pictures agreed to on the figure card: `None` on the planning pass, `[]`
    #: when declined. Only the report draws them.
    figures_plan: list[dict] | None = None,
    #: Draws the figures; the image default, not the writer.
    image_model: dict | None = None,
    project_sources: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Drives one report to completion and settles it. The document is an artifact with versions."""
    sections: list[dict] = []
    proposal: dict | None = None
    questions: list[dict] | None = None
    usage = {"inputTokens": 0, "outputTokens": 0}
    #: From the outline step; empty when none was given.
    doc_title = ""
    #: The shelf the sections cited from, numbered as the prose refers to.
    sources: list[dict] = []
    research_log: dict[str, Any] | None = None

    if routing:
        # First, so the model badge updates.
        yield chat_service.sse({"type": "privacy_route", **routing})
    if skills_event:
        yield chat_service.sse(skills_event)
    for step in context_steps or ():
        yield chat_service.sse(_step_event(step))
    try:
        stream = report_service.write(
            web_search=web_search,
            figures_plan=figures_plan,
            image_model=image_model,
            request=request,
            approved_plan=approved_plan,
            may_ask=may_ask,
            model=model["id"],
            outline_model=(outline_model or {}).get("id", ""),
            api_key=api_key,
            trusted_context=trusted_context,
            untrusted_context=untrusted_context,
            project_sources=project_sources,
        )
        async for event in stream:
            if event["type"] in ("proposal", "needs"):
                # Stopped on purpose: forwarded, and stored below instead of an artifact.
                if event["type"] == "proposal":
                    proposal = event["plan"]
                else:
                    questions = event["questions"]
                yield chat_service.sse(event)
                continue
            if event["type"] == "report":
                sections = event["sections"]
                continue
            if event["type"] == "sources":
                sources = list(event.get("sources") or [])
            if event["type"] == "research":
                research_log = dict(event.get("research") or {})
            if event["type"] == "title":
                doc_title = str(event.get("title") or "").strip()
            if event["type"] == "usage":
                usage = {k: v for k, v in event.items() if k != "type"}
                continue
            yield chat_service.sse(event)
    except Exception as exc:  # noqa: BLE001 — the turn must still settle
        log.exception("report generation crashed for session %s", session_id)
        yield chat_service.sse(_error_event("보고서를 만들지 못했습니다.", exc))

    # Planned or asked: stored on the session, no artifact written.
    if proposal is not None or questions is not None:
        # The figure question is held until the outline is approved.
        drawn = (proposal or {}).pop("figures", None)
        await _settle_plan_turn(
            session_id=session_id,
            user_id=user_id,
            request=request,
            attachments=list(attachments or []),
            answers=dict(answers or {}),
            model=model,
            outline_model=outline_model,
            usage=usage,
            proposal=proposal,
            questions=questions,
            figures={"plan": proposal, **drawn} if drawn and proposal else None,
        )
        yield chat_service.sse({"type": "usage", **usage, "credits": 0})
        yield chat_service.sse({"type": "done"})
        return

    written = [s for s in sections if (s.get("content") or "").strip()]
    credits = (
        0
        if not written
        else charge_for_tokens(model, usage["inputTokens"], usage["outputTokens"])
        # The planner's own tokens at the planner's own price, when one ran.
        + charge_for_tokens(
            outline_model or model,
            usage.get("outlineInputTokens", 0),
            usage.get("outlineOutputTokens", 0),
        )
    )

    artifact_id: str | None = None
    async with SessionLocal() as db:
        session = await db.get(ChatSession, session_id)
        user = await db.get(User, user_id)
        if session is not None and user is not None:
            # Generated title first; the session title reads as the raw prompt.
            title = (doc_title or session.title or request.strip()[:60] or "보고서")[:200]
            if written:
                artifact_design = design_tokens
                if template is not None and template.look:
                    # The template's look outranks the project default and the request.
                    artifact_design = design_service.normalise_tokens(
                        {**(design_tokens or {}), "visualStyle": template.look}
                    )
                elif not artifact_design:
                    requested_style = str(
                        (approved_plan or {}).get("visualStyle")
                        or design_service.visual_style_for(request)
                    )
                    if requested_style != "editorial":
                        artifact_design = design_service.normalise_tokens(
                            {"visualStyle": requested_style}
                        )
                artifact_id = await _store_document(
                    db,
                    session,
                    user_id=user_id,
                    project_id=project_id,
                    kind=ArtifactKind.report,
                    title=title,
                    summary=_regeneration_summary(request),
                    data={
                        **({"templateId": template.id} if template else {}),
                        "sections": [
                            {
                                "id": s["id"],
                                "heading": s["heading"],
                                "level": s.get("level", 1),
                                "status": "done" if (s.get("content") or "").strip() else "pending",
                                "content": s.get("content") or "",
                            }
                            for s in sections
                        ],
                        "sources": sources,
                        # Kept so a later rewrite is held to the same request.
                        "request": request[:4000],
                        **({"research": research_log} if research_log is not None else {}),
                        "lint": lint.wire(lint.check(lint.from_sections(sections))),
                        # Snapshot, as for the deck.
                        **({"design": artifact_design} if artifact_design else {}),
                        "citationStyle": "APA",
                        "wordCount": report_service.word_count(sections),
                    },
                )
                session.artifact_id = artifact_id
                # The proposal has become the document.
                session.pending = None

                db.add(
                    Message(
                        session_id=session_id,
                        role=Role.assistant,
                        content=f"{len(written)}개 섹션으로 보고서를 작성했습니다.",
                        usage={**usage, "credits": credits},
                        model=model["id"],
                        steps=_prelude_steps(skills_event, context_steps) or None,
                        routing=routing,
                    )
                )
                settle(
                    db,
                    user,
                    credits,
                    reason="report.generate",
                    session_id=session_id,
                    model=model["id"],
                )
            session.updated_at = utcnow()
            db.add(session)
            await db.commit()

    if artifact_id:
        yield chat_service.sse({"type": "artifact", "artifactId": artifact_id})
    yield chat_service.sse({"type": "usage", **usage, "credits": credits})
    yield chat_service.sse({"type": "done"})
