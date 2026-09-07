"""Document surfaces ask when material is short, offer an outline, and write only on approval."""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.routers.sessions import _edited_plan
from app.services import deck, grounding, report
from app.services.workspace_context import ContextFile, _excerpt, _file_report


def test_a_person_can_change_the_visual_direction_before_approving():
    stored = {"title": "제안", "visualStyle": "editorial", "sections": ["배경", "결론"]}
    changed = _edited_plan({"visualStyle": "poster", "sections": ["배경", "결론"]}, stored)
    refused = _edited_plan({"visualStyle": "neon-chaos"}, stored)

    assert changed["visualStyle"] == "poster"
    assert refused["visualStyle"] == "editorial"


class _Client:
    """Answers with canned replies and records what was asked."""

    def __init__(self, replies, posts, **_):
        #: By reference: a new client is opened per call and the script must advance.
        self._replies = replies
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, _url, json):
        self._posts.append(json)
        text = self._replies.pop(0) if self._replies else "{}"

        class R:
            status_code = 200

            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {
                    "choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        return R()


@pytest.fixture
def gateway(monkeypatch):
    async def litellm_config():
        return "http://mock-litellm", "unused"

    for module in (deck, report):
        monkeypatch.setattr(module.settings_store, "litellm_config", litellm_config)
    return monkeypatch


_PLAN = (
    '{"title": "제목", "slides": ['
    '{"title": "제목", "layout": "title"},'
    '{"title": "가", "layout": "bullets"},'
    '{"title": "나", "layout": "two-column"},'
    '{"title": "다", "layout": "quote"}]}'
)


# ── the offer ──────────────────────────────────────────────────────────


async def test_a_deck_stops_at_its_outline_and_writes_nothing(gateway):
    posts: list[dict] = []
    gateway.setattr(deck.httpx, "AsyncClient", lambda **kw: _Client([_PLAN], posts, **kw))

    events = [e async for e in deck.write(request="발표", model="m", api_key="k")]

    kinds = [e["type"] for e in events]
    assert "proposal" in kinds
    # Nothing is written on an unapproved pass.
    assert "deck" not in kinds
    assert "slide" not in kinds
    # Only the planning call was made.
    assert len(posts) == 1


async def test_the_approved_outline_is_what_gets_written(gateway):
    posts: list[dict] = []
    replies = [_PLAN, *['{"bullets":["가"],"notes":""}'] * 6]
    gateway.setattr(deck.httpx, "AsyncClient", lambda **kw: _Client(replies, posts, **kw))

    offered = [e async for e in deck.write(request="발표", model="m", api_key="k")]
    plan = next(e["plan"] for e in offered if e["type"] == "proposal")

    written = [
        e async for e in deck.write(request="발표", model="m", api_key="k", approved_plan=plan)
    ]

    slides = next(e["slides"] for e in written if e["type"] == "deck")
    assert [s["title"] for s in slides] == [item["title"] for item in plan["slides"]]
    # Planned once; the approved plan is not re-planned.
    assert sum(1 for p in posts if "발표 슬라이드의 제목과 구성" in str(p)) == 1


async def test_an_empty_approval_writes_nothing(gateway):
    posts: list[dict] = []
    gateway.setattr(deck.httpx, "AsyncClient", lambda **kw: _Client([], posts, **kw))

    events = [
        e
        async for e in deck.write(
            request="발표", model="m", api_key="k", approved_plan={"slides": []}
        )
    ]

    assert [e["type"] for e in events] == ["error", "usage"]
    assert posts == []


async def test_a_report_offers_its_headings_the_same_way(gateway):
    posts: list[dict] = []
    replies = ['{"title": "제목", "sections": ["가", "나", "다", "라"]}']
    gateway.setattr(report.httpx, "AsyncClient", lambda **kw: _Client(replies, posts, **kw))

    events = [e async for e in report.write(request="보고서", model="m", api_key="k")]

    proposal = next(e for e in events if e["type"] == "proposal")
    assert proposal["plan"]["sections"] == ["가", "나", "다", "라"]
    assert "section" not in [e["type"] for e in events]


# ── the question ───────────────────────────────────────────────────────


def test_a_file_that_arrived_short_is_worth_stopping_over():
    short = grounding.file_shortfalls(
        (
            ContextFile("논문.pdf", "truncated", 24_000, 74_200),
            ContextFile("표.csv", "included", 400, 400),
        )
    )

    assert [f.name for f in short] == ["논문.pdf"]
    question = grounding.questions_for(short)[0]
    # The numbers are in the question.
    assert "74,200" in question.detail
    assert "24,000" in question.detail


def test_a_page_and_a_half_off_the_end_is_not_worth_stopping_over():
    """A small truncation does not trigger a question."""
    assert grounding.file_shortfalls((ContextFile("서식.docx", "truncated", 9_500, 10_000),)) == []


def test_an_unreadable_file_asks_a_different_question():
    questions = grounding.questions_for([ContextFile("스캔본.pdf", "unreadable", 0, 0)])

    assert [q.id for q in questions] == ["unreadable"]
    assert "OCR" in questions[0].detail


def test_the_outline_call_may_ask_instead_of_planning(gateway):
    """The outline call may return a question instead of a plan."""
    asked = '{"needs": [{"question": "어느 실험을 다룰까요?", "options": ["전체", "3장만"]}]}'

    parsed = grounding.parse_needs(asked)

    assert [q.question for q in parsed] == ["어느 실험을 다룰까요?"]
    assert parsed[0].options == ["전체", "3장만"]
    # An ordinary outline is not a question.
    assert grounding.parse_needs(_PLAN) is None


async def test_a_deck_that_asks_writes_nothing_either(gateway):
    posts: list[dict] = []
    asked = '{"needs": [{"question": "무엇을 근거로 할까요?", "options": []}]}'
    gateway.setattr(deck.httpx, "AsyncClient", lambda **kw: _Client([asked], posts, **kw))

    events = [e async for e in deck.write(request="이 논문으로 발표", model="m", api_key="k")]

    kinds = [e["type"] for e in events]
    assert "needs" in kinds
    assert "proposal" not in kinds
    assert "deck" not in kinds


async def test_going_with_what_was_read_is_not_asked_again(gateway):
    """After 있는 자료로 진행 the planner does not ask again."""
    posts: list[dict] = []
    gateway.setattr(deck.httpx, "AsyncClient", lambda **kw: _Client([_PLAN], posts, **kw))

    events = [
        e
        async for e in deck.write(
            request="이 논문으로 발표", model="m", api_key="k", may_ask=False
        )
    ]

    kinds = [e["type"] for e in events]
    assert "needs" not in kinds, "물음이 다시 나왔습니다"
    assert "proposal" in kinds
    # The answer is folded into the prompt, not filtered on the way back.
    sent = "\n".join(str(m) for post in posts for m in post.get("messages", []))
    assert "있는 자료로 진행" in sent
    assert "되물어라" not in sent


async def test_an_unreadable_outline_is_asked_for_once_more(gateway):
    """An unparsable outline is retried once."""
    posts: list[dict] = []
    unreadable = "구성은 다음과 같습니다:\n```\n(설명만 있고 JSON 이 없음)\n```"
    # Shared across clients so the retry sees the second reply.
    replies = [unreadable, _PLAN]
    gateway.setattr(deck.httpx, "AsyncClient", lambda **kw: _Client(replies, posts, **kw))

    events = [e async for e in deck.write(request="가상환경 발표", model="m", api_key="k")]

    kinds = [e["type"] for e in events]
    assert "error" not in kinds, "한 번 더 물어보지 않고 포기했습니다"
    assert "proposal" in kinds
    assert len(posts) == 2, "재시도가 일어나지 않았습니다"


async def test_an_outline_unreadable_twice_still_gives_up(gateway):
    """An outline unparsable twice gives up."""
    posts: list[dict] = []
    replies = ["설명뿐, JSON 없음", "여전히 설명뿐"]
    gateway.setattr(deck.httpx, "AsyncClient", lambda **kw: _Client(replies, posts, **kw))

    events = [e async for e in deck.write(request="가상환경 발표", model="m", api_key="k")]

    assert [e["type"] for e in events].count("error") == 1
    assert len(posts) == 2


# ── answering has to change something ──────────────────────────────────


def test_the_answer_moves_the_excerpt_to_the_part_that_was_asked_for():
    """A focus answer moves the excerpt to the named part."""
    text = "머리말 " * 500 + "평가 결과 는 다음과 같다 " * 200 + "맺음말 " * 500

    head = _excerpt(text, 1_000, focus="")
    aimed = _excerpt(text, 1_000, focus="평가 결과")

    assert "평가 결과" not in head
    assert "평가 결과" in aimed
    # The excerpt says its beginning is missing.
    assert "앞" in aimed and "생략" in aimed


def test_a_word_on_every_page_does_not_outvote_the_one_that_was_asked_for():
    """「제290조는 무슨 조항이야」: 조항 is everywhere, 제290조 once, and once wins."""
    article = "제{n}조(일반 사항) 이 조항은 원칙을 정한다. " + "세부 절차는 지침에 따른다. " * 6
    text = "합성 문서\n" + "\n".join(article.format(n=n) for n in range(1, 61))

    aimed = _excerpt(text, 2_000, focus="제50조는 무슨 조항이야? 문서에 있는 그대로.")

    assert "제50조(" in aimed
    assert "제1조(" not in aimed


def test_a_focus_nothing_matches_falls_back_to_the_beginning():
    text = "가나다라마바사" * 500

    assert _excerpt(text, 100, focus="없는말") == text[:100]


def test_going_with_what_was_read_is_not_a_focus():
    """있는 자료로 진행 does not move the excerpt."""
    assert grounding.focus_terms({"focus": "읽은 앞부분만으로 진행"}) == ""
    assert grounding.focus_terms({"focus": "평가 결과"}) == "평가 결과"


def test_answers_are_added_to_the_request_and_never_replace_it():
    merged = grounding.merge_answers("이 논문으로 발표자료", {"focus": "4장 평가만"})

    assert merged.startswith("이 논문으로 발표자료")
    assert "4장 평가만" in merged


def test_the_budget_is_what_it_always_was():
    """The excerpt budget stays 24,000 characters."""
    assert settings.file_context_chars == 24_000


# ── what the model is told about the files ─────────────────────────────


def test_the_model_is_told_what_became_of_every_file():
    """The prompt reports what became of every attachment."""
    report = _file_report(
        (
            ContextFile("논문.pdf", "truncated", 24_000, 74_200),
            ContextFile("스캔본.pdf", "unreadable", 0, 0),
            ContextFile("표.csv", "included", 400, 400),
        ),
        (),
    )

    assert "74,200" in report and "24,000" in report
    assert "OCR" in report
    # The complete file is listed too.
    assert "표.csv" in report
    assert "추측하지 말고" in report
    assert "받지 못했다고 말해서는 안 된다" in report


def test_a_truncated_file_is_told_not_to_fill_in_the_rest():
    report = _file_report((ContextFile("논문.pdf", "truncated", 100, 900),), ())

    assert "지어내지 마라" in report


def test_no_files_means_no_report():
    """No attachments, no attachment report."""
    assert _file_report((), ()) == ""


def test_the_report_is_trusted_context_not_reference_data():
    """The attachment report is placed as trusted context, not reference data."""
    from app.services.workspace_context import ContextBlock

    block = ContextBlock(
        "files.report", _file_report((ContextFile("a", "included", 1, 1),), ()), True
    )

    assert block.trusted is True


def test_approving_is_not_folded_into_the_request_as_a_condition():
    """이대로 생성 does not reach the writing prompts or the history."""
    from app.routers.sessions import _regeneration_summary

    merged = grounding.merge_answers("전이학습 발표자료", {"focus": "의료 영상 사례"})

    assert _regeneration_summary(merged) == "재생성 전 · 전이학습 발표자료"


def test_a_request_built_on_an_attachment_that_never_arrived_asks_instead_of_writing():
    """A request citing an attachment that never arrived asks instead of writing."""
    gap = grounding.missing_attachment("첨부한 내용을 바탕으로 보고서 작성해줘", ())

    assert gap is not None
    assert gap.id == "no_attachment"
    # Proceeding without the file must be one of the options.
    assert any("진행" in option for option in gap.options)


def test_the_attachment_gap_is_the_act_of_attaching_not_the_word():
    """A request about attachments as a subject is not a missing attachment."""
    assert grounding.missing_attachment("이메일 첨부 파일 관리 정책 보고서를 써 줘", ()) is None
    assert grounding.missing_attachment("전이학습 발표자료 만들어 줘", ()) is None


def test_a_request_naming_an_attachment_that_did_arrive_is_not_asked_about():
    """A request naming an attachment that arrived is not asked about."""
    assert (
        grounding.missing_attachment(
            "첨부한 계획서로 보고서 써 줘",
            (ContextFile("계획서.hwpx", "included", 4_974, 4_974),),
        )
        is None
    )
