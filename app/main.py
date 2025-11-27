import asyncio
import base64
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yomitoku.data.functions import load_image, load_pdf
from yomitoku.document_analyzer import DocumentAnalyzer
from yomitoku.export.export_html import convert_html
from yomitoku.utils.logger import set_logger

logger = set_logger(__name__, "INFO")

DEFAULT_THREADS = max(1, os.cpu_count() or 4)
MAX_PARALLEL = max(2, os.cpu_count() or 4)


def resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def clamp_parallelism(value: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, MAX_PARALLEL))


def configure_torch_threads(threads: int) -> int:
    try:
        threads = int(threads)
    except (TypeError, ValueError):
        threads = DEFAULT_THREADS

    threads = max(1, threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads // 2))
    except Exception:
        # Some backends may not expose interop threads (e.g., minimal builds)
        pass
    return threads


def normalize_text(text: str, collapse_line_breaks: bool) -> str:
    text = (text or "").strip()
    if collapse_line_breaks:
        text = " ".join(text.splitlines())
    return text


def build_blocks(doc, collapse_line_breaks: bool = False) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    paragraphs = sorted(
        doc.paragraphs, key=lambda p: p.order if p.order is not None else 0
    )
    for idx, paragraph in enumerate(paragraphs):
        blocks.append(
            {
                "id": f"p-{idx}",
                "type": "paragraph",
                "order": paragraph.order if paragraph.order is not None else idx,
                "text": normalize_text(paragraph.contents, collapse_line_breaks),
                "direction": paragraph.direction,
                "role": paragraph.role,
                "box": paragraph.box,
            }
        )

    tables = sorted(doc.tables, key=lambda t: t.order if t.order is not None else 0)
    for t_idx, table in enumerate(tables):
        cells = sorted(table.cells, key=lambda c: (c.row, c.col))
        for cell in cells:
            blocks.append(
                {
                    "id": f"t{t_idx}-r{cell.row}c{cell.col}",
                    "type": "table-cell",
                    "order": table.order if table.order is not None else t_idx,
                    "text": normalize_text(cell.contents, collapse_line_breaks),
                    "row": cell.row,
                    "col": cell.col,
                    "box": cell.box,
                }
            )

    figures = sorted(doc.figures, key=lambda f: f.order if f.order is not None else 0)
    for f_idx, fig in enumerate(figures):
        blocks.append(
            {
                "id": f"fig-{f_idx}",
                "type": "figure",
                "order": fig.order if fig.order is not None else f_idx,
                "text": "",
                "box": fig.box,
            }
        )

    blocks = sorted(blocks, key=lambda b: b["order"])
    return blocks


def join_blocks(blocks: List[Dict[str, Any]], exclude_ids: Set[str]) -> str:
    lines: List[str] = []
    for block in blocks:
        if block["id"] in exclude_ids:
            continue
        text = block.get("text") or ""
        if not text:
            continue
        if block["type"] == "table-cell":
            prefix = f"[Table r{block.get('row')} c{block.get('col')}] "
            lines.append(prefix + text)
        else:
            lines.append(text)
    return "\n".join(lines)


class EngineSettings(BaseModel):
    device: str = Field("auto", description="cpu/cuda/auto")
    use_onnx: bool = Field(False, description="Use ONNX Runtime when possible")
    visualize: bool = Field(True, description="Render OCR/layout overlays")
    parallelism: int = Field(2, description="ThreadPool workers for inference")
    torch_threads: int = Field(DEFAULT_THREADS, description="Torch CPU threads")
    dpi: int = Field(200, description="PDF rasterization DPI")
    collapse_line_breaks: bool = Field(
        False, description="Remove line breaks inside paragraphs and cells"
    )

    @property
    def cache_key(self):
        return (
            self.device,
            self.use_onnx,
            self.visualize,
            self.parallelism,
        )


class TaskRunner:
    def __init__(self, workers: int):
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self.workers = workers

    async def resize(self, workers: int):
        workers = clamp_parallelism(workers)
        async with self._lock:
            if workers == self.workers:
                return

            self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=workers)
            self.workers = workers

    async def run(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)


class AnalyzerManager:
    def __init__(self):
        self._cache: Dict[tuple, DocumentAnalyzer] = {}
        self._lock = asyncio.Lock()

    async def get(self, settings: EngineSettings) -> DocumentAnalyzer:
        key = settings.cache_key
        if key in self._cache:
            return self._cache[key]

        async with self._lock:
            if key in self._cache:
                return self._cache[key]

            device = resolve_device(settings.device)
            configs = {
                "ocr": {
                    "text_detector": {
                        "device": device,
                        "visualize": settings.visualize,
                        "infer_onnx": settings.use_onnx,
                    },
                    "text_recognizer": {
                        "device": device,
                        "visualize": settings.visualize,
                        "infer_onnx": settings.use_onnx,
                    },
                },
                "layout_analyzer": {
                    "layout_parser": {
                        "device": device,
                        "visualize": settings.visualize,
                        "infer_onnx": settings.use_onnx,
                    },
                    "table_structure_recognizer": {
                        "device": device,
                        "visualize": settings.visualize,
                        "infer_onnx": settings.use_onnx,
                    },
                },
            }

            analyzer = DocumentAnalyzer(
                configs=configs,
                device=device,
                visualize=settings.visualize,
                split_text_across_cells=True,
                max_workers=settings.parallelism,
            )
            self._cache[key] = analyzer
            return analyzer


def encode_image(img):
    if img is None:
        return None

    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        return None

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def aggregate_text(doc, collapse_line_breaks: bool = False) -> str:
    blocks = build_blocks(doc, collapse_line_breaks=collapse_line_breaks)
    return join_blocks(blocks, exclude_ids=set())


def build_html_preview(doc, collapse_line_breaks: bool = False) -> str:
    html_str, _ = convert_html(
        doc,
        out_path="",
        ignore_line_break=collapse_line_breaks,
        export_figure=False,
        export_figure_letter=False,
        img=None,
    )
    return html_str


def load_pages(path: Path, dpi: int):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(str(path), dpi=dpi)
    return load_image(str(path))


async def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload.read()
        tmp.write(content)
        return Path(tmp.name)


def run_analysis(
    analyzer: DocumentAnalyzer, pages: List[Any], settings: EngineSettings
) -> Dict[str, Any]:
    results = []
    for idx, img in enumerate(pages):
        doc, ocr_vis, layout_vis = analyzer(img)
        blocks = build_blocks(doc, collapse_line_breaks=settings.collapse_line_breaks)
        results.append(
            {
                "page": idx + 1,
                "text": join_blocks(blocks, exclude_ids=set()),
                "html_preview": build_html_preview(
                    doc, collapse_line_breaks=settings.collapse_line_breaks
                ),
                "summary": {
                    "paragraphs": len(doc.paragraphs),
                    "tables": len(doc.tables),
                    "figures": len(doc.figures),
                    "words": len(doc.words),
                },
                "json": doc.model_dump(),
                "blocks": blocks,
                "ocr_preview": encode_image(ocr_vis),
                "layout_preview": encode_image(layout_vis),
            }
        )

    return {
        "device": resolve_device(settings.device),
        "use_onnx": settings.use_onnx,
        "visualize": settings.visualize,
        "parallelism": settings.parallelism,
        "torch_threads": settings.torch_threads,
        "page_count": len(pages),
        "collapse_line_breaks": settings.collapse_line_breaks,
        "results": results,
    }


app = FastAPI(
    title="YomiToku OCR Server",
    description="LAN-ready OCR API powered by YomiToku",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

ANALYZER_MANAGER = AnalyzerManager()
RUNNER = TaskRunner(workers=2)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = static_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=500, detail="UI not found. Ensure app/static/index.html exists."
        )

    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def status():
    return {
        "cuda_available": torch.cuda.is_available(),
        "current_device": "cuda" if torch.cuda.is_available() else "cpu",
        "threads": torch.get_num_threads(),
        "max_parallel": MAX_PARALLEL,
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    device: str = Form("auto"),
    use_onnx: bool = Form(False),
    visualize: bool = Form(True),
    parallelism: int = Form(2),
    torch_threads: int = Form(DEFAULT_THREADS),
    dpi: int = Form(200),
    collapse_line_breaks: bool = Form(False),
):
    settings = EngineSettings(
        device=device,
        use_onnx=use_onnx,
        visualize=visualize,
        parallelism=clamp_parallelism(parallelism),
        torch_threads=torch_threads,
        dpi=dpi,
        collapse_line_breaks=collapse_line_breaks,
    )

    settings.torch_threads = configure_torch_threads(settings.torch_threads)
    await RUNNER.resize(settings.parallelism)

    tmp_path = await save_upload(file)

    try:
        pages = load_pages(tmp_path, settings.dpi)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        logger.error("Failed to load input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analyzer = await ANALYZER_MANAGER.get(settings)

    try:
        payload = await RUNNER.run(run_analysis, analyzer, pages, settings)
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return payload
