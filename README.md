日本語版 | [English](README_EN.md)

# YomiToku OCR / Document AI

日本語文書に特化した OCR・レイアウト解析エンジンです。CLI と FastAPI サーバー、軽量モデルを備え、GPU なしでも動作可能です。

[ドキュメント](https://kotaro-kinoshita.github.io/yomitoku/)｜[PyPI](https://pypi.org/project/yomitoku/)｜[サンプル出力](gallery.md)

## 特長
- 4 つの独立モデル（検出・認識・レイアウト・表構造）を日本語データで学習し、縦書き・手書きにも対応
- 出力形式: HTML / Markdown / JSON / CSV / searchable PDF、図表抽出も対応
- 軽量モデル（`--lite`）で CPU でも高速、通常モデルは GPU で高精度
- 1 コマンドで可視化済み画像とテキストをまとめて取得

## クイックスタート
### インストール
```bash
pip install yomitoku
# CUDA 環境の場合は PyTorch を環境に合わせて先にインストールしてください
```

### CLI で変換
```bash
yomitoku path/to/image_or_dir -f md -o results -v --figure
```
- まとめて出力: `--combine`（PDF 複数ページを結合）
- 軽量モデル: `--lite -d cpu`
- 段落内改行を除去: `--ignore_line_break`

### Web UI / API サーバー
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
ブラウザで `http://localhost:8000` を開くとコントロールパネルが使えます。設定は UI の「詳細設定」モーダルから変更できます。

## 主な CLI オプション
| オプション | 説明 |
| --- | --- |
| `-f, --format {json,csv,html,md,pdf}` | 出力フォーマット |
| `-o, --outdir DIR` | 出力先ディレクトリ（なければ作成） |
| `-d, --device {cuda,cpu,auto}` | 推論デバイス |
| `-l, --lite` | 軽量モデルで高速推論 |
| `--vis, -v` | 検出・レイアウトを可視化した画像を出力 |
| `--ignore_line_break` | 段落内の改行を無視して連結 |
| `--figure` / `--figure_letter` | 図表画像・図表内文字を抽出 |
| `--combine` | PDF 複数ページを 1 ファイルに結合 |

ヘルプ: `yomitoku --help`

## 動作条件
- Python 3.10–3.12
- CUDA 11.8 以上推奨（CPU でも動作可）
- VRAM 8GB 以内で運用可能、軽量モデルなら低リソースで可

## 開発・テスト
```bash
pip install -e ".[mcp]" pytest ruff
pytest tests -q
ruff check src tests
```

## ライセンスと商用利用
ソースコードおよび関連モデルは **CC BY-NC-SA 4.0** で提供されています。非商用の個人利用・研究利用は自由に行えます。  
商用利用は製品版ライセンス（オンプレ/AWS Marketplace）をご利用ください。詳細は [docs/commercial_use_guideline.ja.md](docs/commercial_use_guideline.ja.md) を参照してください。
