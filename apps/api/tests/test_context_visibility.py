"""Timeline steps reporting memories, attachments and project knowledge fed into a turn."""

from __future__ import annotations

from app.core.config import settings
from app.models.chat import ChatSession, Message, Role, SessionKind
from app.models.user import User, UserStatus, utcnow
from app.models.workspace import Memory, Project, StoredFile
from app.routers import sessions as sessions_router
from app.routers.sessions import _context_steps, _prelude_steps, _step_event
from app.services.workspace_context import assemble

# ── memories ───────────────────────────────────────────────────────────


async def test_the_turn_names_the_memories_it_answered_from_and_quotes_none():
    """The memories step names memories and never carries their bodies."""
    db = _Db(memories=[_memory("소속", "단국대학교"), _memory("말투", "존댓말을 쓴다")])
    context = await assemble(db, _user(), _session())

    assert set(context.loaded_memories) == {"말투", "소속"}
    assert context.total_memories == 2
    step = _one(_context_steps(context), "context-memories")
    assert step["label"] == "메모리 2건 참고"
    assert "단국대학교" not in repr(step)
    # The bodies go to the model in the block only.
    assert "단국대학교" in next(b.text for b in context.blocks if b.source == "memory")


async def test_a_turn_that_could_only_carry_some_of_them_says_which_some():
    """A partial memory load reports loaded-of-total."""
    db = _Db(memories=[_memory(f"사실 {i:02d}", "본문") for i in range(45)])
    context = await assemble(db, _user(), _session())

    assert len(context.loaded_memories) == 40
    assert context.total_memories == 45
    step = _one(_context_steps(context), "context-memories")
    assert step["label"] == "메모리 40건 참고"
    assert step["detail"].endswith("외 34건 · 저장된 45건 중 최근 40건")


async def test_a_turn_with_nothing_remembered_adds_no_line():
    """No memories means no memories step."""
    context = await assemble(_Db(), _user(), _session())
    assert _context_steps(context) == []


# ── attachments ────────────────────────────────────────────────────────


async def test_each_attached_file_reports_what_survived_the_budget(monkeypatch):
    """The char budget is spent in attachment order; each file reports its own state."""
    monkeypatch.setattr(settings, "file_context_chars", 30)
    files = [
        _file("첫째.txt", "가" * 20),
        _file("둘째.txt", "나" * 40),
        _file("셋째.txt", "다" * 10),
    ]
    context = await _assembled_with(files)

    assert [(f.name, f.state) for f in context.attachments] == [
        ("첫째.txt", "included"),
        ("둘째.txt", "truncated"),
        ("셋째.txt", "omitted"),
    ]
    assert context.attachments[1].kept_chars == 10
    assert context.attachments[1].total_chars == 40

    step = _one(_context_steps(context), "context-attachments")
    assert step["label"] == "첨부 3개 중 1개 잘림, 1개 빠짐"
    assert step["detail"] == "둘째.txt 10자만 반영 · 셋째.txt 분량을 넘겨 제외"
    assert step["files"][2] == {
        "name": "셋째.txt",
        "state": "omitted",
        "keptChars": 0,
        "totalChars": 10,
    }


async def test_a_file_that_fit_whole_still_gets_its_line():
    context = await _assembled_with([_file("메모.txt", "짧다")])
    step = _one(_context_steps(context), "context-attachments")
    assert step["label"] == "첨부 1개 반영"
    assert step["detail"] == "메모.txt"


async def test_a_file_nothing_could_be_read_out_of_is_reported_where_it_was_attached():
    """Attachments are reported in attachment order, unreadable ones included."""
    files = [_file("보고서.pdf", "", error="스캔본"), _file("메모.txt", "읽힌다")]
    context = await _assembled_with(files)

    assert [(f.name, f.state) for f in context.attachments] == [
        ("보고서.pdf", "unreadable"),
        ("메모.txt", "included"),
    ]
    step = _one(_context_steps(context), "context-attachments")
    assert step["label"] == "첨부 2개 중 1개 빠짐"
    assert step["detail"] == "보고서.pdf 읽지 못함"


# ── earlier turns' attachments ─────────────────────────────────────────


async def test_a_file_attached_earlier_in_the_conversation_is_still_read(monkeypatch):
    """An upload from an earlier turn reaches later turns, after this turn's own files."""
    monkeypatch.setattr(settings, "file_context_chars", 30)
    earlier = _file("학칙.pdf", "가" * 20)
    earlier.session_id = "session-1"
    now = _file("메모.txt", "나" * 10)
    context = await assemble(
        _Db(files=[earlier, now]), _user(), _session(), attachment_ids=[now.id]
    )

    assert [(f.name, f.state) for f in context.attachments] == [("메모.txt", "included")]
    assert [(f.name, f.state) for f in context.carried] == [("학칙.pdf", "included")]
    sources = [block.source for block in context.blocks]
    assert sources.index("attachment") < sources.index("attachment.earlier")
    report = next(block.text for block in context.blocks if block.source == "files.report")
    assert "학칙.pdf (앞선 턴에 첨부) — 전체 20자 전달됨" in report

    step = _one(_context_steps(context), "context-earlier")
    assert step["label"] == "이전 첨부 1개 반영"
    assert step["detail"] == "학칙.pdf"


async def test_an_earlier_file_too_long_to_carry_whole_is_excerpted_around_the_question(
    monkeypatch,
):
    """Once a carried file no longer fits, the part the question is about is what goes."""
    monkeypatch.setattr(settings, "file_context_chars", 1_000)
    earlier = _file("학칙.pdf", "가" * 3_000 + "제80조 휴학은 두 해까지 한다." + "나" * 3_000)
    earlier.session_id = "session-1"
    # 「제80조는」: the particle is the question's, not the document's.
    context = await assemble(
        _Db(files=[earlier]), _user(), _session(), question="제80조는 무슨 조항이야?"
    )

    block = next(block.text for block in context.blocks if block.source == "attachment.earlier")
    assert "제80조 휴학은 두 해까지 한다." in block
    assert [(f.name, f.state) for f in context.carried] == [("학칙.pdf", "truncated")]


async def test_an_earlier_file_the_budget_cannot_carry_is_still_listed_as_arrived(monkeypatch):
    """This turn's file spends the budget; the earlier one is named so it is not denied."""
    monkeypatch.setattr(settings, "file_context_chars", 10)
    earlier = _file("학칙.pdf", "가" * 50)
    earlier.session_id = "session-1"
    now = _file("메모.txt", "나" * 10)
    context = await assemble(
        _Db(files=[earlier, now]), _user(), _session(), attachment_ids=[now.id]
    )

    assert [(f.name, f.state) for f in context.carried] == [("학칙.pdf", "omitted")]
    assert "attachment.earlier" not in [block.source for block in context.blocks]
    report = next(block.text for block in context.blocks if block.source == "files.report")
    assert "학칙.pdf (앞선 턴에 첨부) — 분량 때문에" in report
    assert _one(_context_steps(context), "context-earlier")["label"] == "이전 첨부 1개 중 1개 빠짐"


async def test_another_conversations_upload_is_not_carried():
    other = _file("학칙.pdf", "가" * 20)
    other.session_id = "session-2"
    context = await assemble(_Db(files=[other]), _user(), _session())

    assert context.carried == ()
    assert "attachment.earlier" not in [block.source for block in context.blocks]


# ── project knowledge ──────────────────────────────────────────────────


async def test_project_knowledge_dropped_for_size_gets_the_same_line(monkeypatch):
    """Project knowledge reports truncation and omission under its own step id."""
    monkeypatch.setattr(settings, "file_context_chars", 5)
    project = Project(id="project-1", user_id="user-1", name="연구")
    db = _Db(project=project, files=[_file("규정.md", "가" * 50), _file("연혁.md", "나" * 50)])
    context = await assemble(db, _user(), _session(project_id=project.id))

    assert [(f.name, f.state) for f in context.knowledge] == [
        ("규정.md", "truncated"),
        ("연혁.md", "omitted"),
    ]
    step = _one(_context_steps(context), "context-knowledge")
    assert step["label"] == "프로젝트 지식 2개 중 1개 잘림, 1개 빠짐"
    assert step["detail"] == "규정.md 5자만 반영 · 연혁.md 분량을 넘겨 제외"


async def test_large_project_shelf_selects_relevant_passages_instead_of_oldest_file(monkeypatch):
    monkeypatch.setattr(settings, "file_context_chars", 1_200)
    project = Project(id="project-1", user_id="user-1", name="연구")
    files = [
        _file("먼저 올린 무관 자료.md", ("복지 제도와 휴가 규정 안내입니다. " * 120)),
        _file(
            "배터리 실험.pdf",
            ("[페이지 7]\n고체전해질 배터리의 이온전도도 측정 결과입니다. " * 80),
        ),
    ]
    context = await assemble(
        _Db(project=project, files=files),
        _user(),
        _session(project_id=project.id),
        focus="고체전해질 배터리 이온전도도 결과",
    )

    assert [(f.name, f.state) for f in context.knowledge] == [
        ("먼저 올린 무관 자료.md", "omitted"),
        ("배터리 실험.pdf", "truncated"),
    ]
    selected = context.knowledge[1]
    assert selected.locations == ("7쪽",)
    block = next(b.text for b in context.blocks if b.source == "project.knowledge")
    assert "이온전도도" in block
    assert "복지 제도" not in block


async def test_the_prompt_the_model_reads_is_unchanged_by_the_reporting(monkeypatch):
    """The prompt block still carries its own truncation and omission notices."""
    monkeypatch.setattr(settings, "file_context_chars", 10)
    context = await _assembled_with([_file("긴글.txt", "가" * 30), _file("남은글.txt", "나" * 5)])

    block = next(b.text for b in context.blocks if b.source == "attachment")
    # An excerpt may omit the beginning too, so the notice does not name the tail.
    assert "…(전체 30자 중 일부입니다)" in block
    assert "## 포함되지 않은 파일\n남은글.txt" in block


# ── how the line travels ───────────────────────────────────────────────


def test_a_step_is_stored_by_category_and_sent_by_event_name():
    """Stored steps keep the category in `type`; wire events use `type: step` + `category`."""
    stored = {"id": "context-memories", "type": "thinking", "label": "메모리 1건 참고"}
    wire = _step_event(stored)

    assert wire["type"] == "step"
    assert wire["category"] == "thinking"
    assert stored["type"] == "thinking"


def test_the_turn_opens_with_what_it_was_given_skills_first():
    """Prelude steps are skills first, then context lines."""
    skills = {
        "type": "skills_applied",
        "skills": [{"id": "s1", "name": "검증", "catalogKey": None, "estimatedTokens": 12}],
        "estimatedTokens": 12,
    }
    context = [{"id": "context-memories", "type": "thinking", "label": "메모리 1건 참고"}]

    assert [step["id"] for step in _prelude_steps(skills, context)] == [
        "skills-applied",
        "context-memories",
    ]
    assert _prelude_steps(None, None) == []


# ── memories written out of the turn ───────────────────────────────────


async def test_a_written_memory_says_so_on_the_row_it_came_from(monkeypatch):
    """A memory-written step is stored on the answer row as well as returned."""
    answer = Message(id="message-1", session_id="session-1", role=Role.assistant, content="네")
    db = _EnrichDb(answer)
    _patch_enrichment(monkeypatch, db, written=2)

    step = await sessions_router._enrich_memory(
        user_id="user-1",
        session_id="session-1",
        content="네",
        first_user_message="기억해 줘",
        api_key="key",
        model={"id": "model-a"},
        auto_memory=True,
        message_id=answer.id,
    )

    assert step["label"] == "메모리 2건 저장"
    assert answer.steps == [step]
    assert db.commits == 1


async def test_a_turn_that_remembered_nothing_leaves_the_row_alone(monkeypatch):
    answer = Message(id="message-1", session_id="session-1", role=Role.assistant, content="네")
    db = _EnrichDb(answer)
    _patch_enrichment(monkeypatch, db, written=0)

    step = await sessions_router._enrich_memory(
        user_id="user-1",
        session_id="session-1",
        content="네",
        first_user_message="안녕",
        api_key="key",
        model={"id": "model-a"},
        auto_memory=True,
        message_id=answer.id,
    )

    assert step is None
    assert answer.steps is None


# ── fakes ──────────────────────────────────────────────────────────────


def _one(steps: list[dict], step_id: str) -> dict:
    matched = [step for step in steps if step["id"] == step_id]
    assert len(matched) == 1, [step["id"] for step in steps]
    return matched[0]


async def _assembled_with(files: list[StoredFile]):
    return await assemble(
        _Db(files=files), _user(), _session(), attachment_ids=[f.id for f in files]
    )


class _Result:
    def __init__(self, rows: list):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _Db:
    """In-memory db for `assemble`."""

    def __init__(
        self,
        *,
        memories: list[Memory] | None = None,
        files: list[StoredFile] | None = None,
        project: Project | None = None,
    ):
        self.memories = memories or []
        self.files = files or []
        self.project = project

    async def get(self, model, row_id):
        if model is Project and self.project is not None and row_id == self.project.id:
            return self.project
        return None

    async def exec(self, query):
        table = query.get_final_froms()[0].name
        if table == "memories":
            return _Result(self.memories)
        if table == "files":
            return _Result(self.files)
        if table == "skills":
            return _Result([])
        raise AssertionError(f"unexpected query: {query}")


class _EnrichDb:
    """In-memory db for `_store_artifacts`/`_enrich_memory`: one message row and a commit count."""

    def __init__(self, message: Message):
        self.message = message
        self.session = ChatSession(id="session-1", user_id="user-1")
        self.user = _user()
        self.commits = 0

    async def get(self, model, row_id):
        if model is Message:
            return self.message if row_id == self.message.id else None
        if model is ChatSession:
            return self.session
        if model is User:
            return self.user
        return None

    def add(self, _row):
        return None

    async def commit(self):
        self.commits += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _patch_enrichment(monkeypatch, db: _EnrichDb, *, written: int) -> None:
    monkeypatch.setattr(sessions_router, "SessionLocal", lambda: db)

    async def store_requested(*_args, **_kwargs):
        return None

    async def extract(*_args, **_kwargs):
        return None

    async def remember(*_args, **_kwargs):
        # `(written, usage)`; billing is covered in test_side_calls_are_billed.py.
        return written, {"inputTokens": 0, "outputTokens": 0}

    async def enrichment_model():
        return "model-a"

    monkeypatch.setattr(sessions_router.artifact_extract, "store_requested", store_requested)
    monkeypatch.setattr(sessions_router.artifact_extract, "extract", extract)
    monkeypatch.setattr(sessions_router.auto_memory_service, "extract", remember)
    monkeypatch.setattr(sessions_router.model_service, "resolve_enrichment_model", enrichment_model)


def _user() -> User:
    return User(
        id="user-1",
        email="user@example.com",
        password_hash="hash",
        name="User",
        status=UserStatus.active,
        monthly_credits=100,
    )


def _session(project_id: str | None = None) -> ChatSession:
    return ChatSession(
        id="session-1", user_id="user-1", kind=SessionKind.chat, project_id=project_id
    )


def _memory(name: str, body: str) -> Memory:
    return Memory(user_id="user-1", name=name, body=body, updated_at=utcnow())


def _file(name: str, text: str, *, error: str | None = None) -> StoredFile:
    return StoredFile(id=f"file-{name}", user_id="user-1", name=name, text=text, error=error)
