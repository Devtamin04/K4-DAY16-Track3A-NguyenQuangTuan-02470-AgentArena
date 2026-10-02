"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re
import unicodedata

from harness.middleware import Middleware

_WS_RE = re.compile(r"\s+")


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            doc = ctx.corpus.get(claim.get("doc_id") or "")
            if doc is not None and _on_a_line(claim["text"], doc.body):
                continue
            source = _source_doc(ctx, claim["text"])
            if source is not None:
                claim["doc_id"] = source
        report["citations"] = sorted(
            {c["doc_id"] for c in claims if isinstance(c, dict) and c.get("doc_id")}
        )
        return report


def _norm(text) -> str:
    """Chuẩn hoá giống scorer (NFC, casefold, gộp khoảng trắng) — chỉ để SO
    SÁNH, không bao giờ ghi ngược vào `claim["text"]`."""
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def _on_a_line(text, body) -> bool:
    """`text` có phải trích nguyên văn MỘT DÒNG của `body` không."""
    needle = _norm(text)
    if not needle or not isinstance(body, str):
        return False
    return any(needle in _norm(line) for line in body.splitlines())


def _source_doc(ctx, text):
    """doc_id của tài liệu ĐÃ QUAN SÁT có một dòng chứa `text`, hoặc None.

    Ưu tiên tài liệu về nguyên vẹn từ một lần fetch sạch; sau đó mới tới
    tài liệu chỉ được nhắc doc_id trong quan sát (vd. kết quả search).
    """
    observed = _norm(ctx.observed_text)
    needle = _norm(text)
    if not needle or needle not in observed:
        return None
    candidates = [doc for doc in ctx.corpus.docs if _on_a_line(text, doc.body)]
    for doc in candidates:
        if _norm(doc.body) in observed:
            return doc.doc_id
    for doc in candidates:
        if doc.doc_id in ctx.observed_text:
            return doc.doc_id
    return None
