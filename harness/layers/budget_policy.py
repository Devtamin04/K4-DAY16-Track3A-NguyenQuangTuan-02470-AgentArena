"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL, RealModel, is_degraded
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from arena.model import FINALIZE_SENTINEL, RealModel, is_degraded
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


DUPLICATE_NOTE = (
    "Lệnh này đã được gọi với đúng tham số đó ở một lượt trước và kết quả nằm ở "
    "phía trên — không tốn thêm ngân sách. Đừng lặp lại: dùng một truy vấn KHÁC "
    "hoặc một tài liệu chưa đọc, hoặc viết FINAL nếu đã đủ bằng chứng."
)


#: Search breadth on a live endpoint. Measured on gpt-oss:120b's own
#: queries: the answer document of the deep briefs ranks 4th-7th, i.e. just
#: outside the k=5 the model asks for. Breadth costs no tool call; kept
#: under 8 so one search cannot trip the runner's dump-signature review flag
#: (>= 8 distinct docs returned with <= 3 model calls).
LIVE_SEARCH_K = 7

#: Characters of snippet kept per hit in a compacted search observation.
COMPACT_SNIPPET_CHARS = 140

EMPTY_SEARCH_NOTE = (
    "(không có kết quả — hãy thử truy vấn ngắn hơn, bỏ dấu ngoặc kép, dùng tên "
    "lĩnh vực chung)"
)


def _live(ctx) -> bool:
    """A real HTTP endpoint (`RealModel`), possibly behind the runner's proxy."""
    model = getattr(ctx, "model", None)
    return isinstance(getattr(model, "inner", model), RealModel)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compact_search(result):
    """One line per hit: doc_id, title, and the snippet minus its first line
    (the corpus-wide document header). Cuts the tokens of a wider search;
    doc_ids and titles — what the model picks fetches from — stay intact.
    Anything that is not a clean JSON hit list passes through untouched."""
    import json

    if not result.ok or not isinstance(result.content, str):
        return result
    try:
        hits = json.loads(result.content)
    except ValueError:
        return result
    if not isinstance(hits, list) or not all(isinstance(h, dict) for h in hits):
        return result
    lines = []
    for hit in hits:
        snippet = str(hit.get("snippet", "")).split("\n")
        tail = " / ".join(part.strip() for part in snippet[1:] if part.strip())
        lines.append(
            f"{hit.get('doc_id', '')} | {hit.get('title', '')} | {tail[:COMPACT_SNIPPET_CHARS]}"
        )
    content = "\n".join(lines) if lines else EMPTY_SEARCH_NOTE
    return ToolResult(ok=True, content=content, error=result.error)


def _call_key(name, args) -> str:
    """Canonical identity of a tool call; search queries compare normalised."""
    import json

    args = dict(args) if isinstance(args, dict) else {}
    if isinstance(args.get("query"), str):
        args["query"] = " ".join(args["query"].casefold().split())
    return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if not self._spent(ctx):
            return messages
        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_tool_call(self, ctx, call, name, args):
        # A call identical to one that already came back clean buys nothing
        # but spends budget — measured on gpt-oss:120b: the same search sent
        # three times and the same doc fetched five, on an 8-call budget.
        # Answer it from memory instead, without touching the tool.
        key = _call_key(name, args)
        done = ctx.state.setdefault("budget_done_calls", set())
        if key in done:
            ctx.state["budget_duplicates"] = ctx.state.get("budget_duplicates", 0) + 1
            return ToolResult(ok=True, content=DUPLICATE_NOTE, error=None)
        if not self._spent(ctx) or name == "submit":
            live = _live(ctx)
            if live and name == "search":
                args = {**args, "k": max(_as_int(args.get("k"), 5), LIVE_SEARCH_K)}
            result = call(name, args)
            if result.ok and not is_degraded(result.content):
                done.add(key)
            if live and name == "search":
                result = _compact_search(result)
            return result
        ctx.state["budget_blocked"] = ctx.state.get("budget_blocked", 0) + 1
        return ToolResult(
            ok=False,
            content="",
            error="Ngân sách công cụ đã hết — hãy trả lời FINAL bằng bằng chứng đang có.",
        )
