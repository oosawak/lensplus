# lensplus

`lensplus` は、Instagram のスクショからレシピ情報と店舗情報を抽出して保存・検索・編集する、ブラウザ完結型のアプリです。

## 目標

- フロントエンドのみで動く
- `docs/index.html` を入口にする
- OCR は `Tesseract.js`
- LLM は `transformers.js`
- 保存先は `IndexedDB`
- 画像はローカル内で Base64 保存

## 主要ファイル

- [docs/index.html](/Users/oosawak/Documents/lensplus/docs/index.html)
- [PROGRESS.md](/Users/oosawak/Documents/lensplus/PROGRESS.md)

## 使い方

1. `docs/index.html` をブラウザで開く、または静的サーバーで配信する
2. 3 枚の画像を選ぶ
3. `画像処理` を押して OCR を実行する
4. 必要なら LLM モデルを読み込む
5. `DB に保存` で IndexedDB に保存する

## モデルについて

- 小さいモデルは Hugging Face から自動取得する
- 大きいモデルは URL を見ながら手動で投入する
- モバイルでは軽量モデルを優先する

## 開発メモ

- 仕様の詳細は、別途渡された設計書を基準にする
- 変更が入ったら `PROGRESS.md` に反映する
- 未完了の作業はここではなく `PROGRESS.md` で管理する

