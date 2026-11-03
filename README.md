# jxl-mini-encoder

8 bit sRGB の RGB / RGBA 入力に対応した、純 Rust の JPEG XL エンコーダです。C/C++ ツールチェーン・CMake・bindgen は不要です。

libjxl のエンコード処理・ヒューリスティクス・定数を元に実装しています。対応機能やパラメータの扱いには独自の変更があり、libjxl と同じ挙動を保証するものではありません。

対象範囲:

- 入力: 8 bit sRGB の RGB / RGBA（ストレートアルファ）
- 非可逆: VarDCT（XYB、適応量子化、DCT8〜DCT64 の可変ブロック、EPF、パッチ）
- 可逆（品質 100）: Modular（RCT 選択、MA 木学習、重み付き予測器）
- effort 1–10、スレッド数指定
- 出力: 裸のコードストリーム（`FF 0A`）。ICC・アニメーション・プログレッシブ・デコーダは対象外

## 使い方

```rust
let image = jxl_mini_encoder::ImageRef::rgba8(width, height, &pixels)?;
let jxl = jxl_mini_encoder::Encoder::new()
    .quality(75.0) // 100 で可逆
    .effort(7)
    .threads(4)
    .encode(&image)?;
```

## パラメータの扱い

範囲外の値に対する処理は libjxl と異なります。libjxl の距離・effort 設定 API が範囲外の値をエラーにするのに対し、このライブラリでは以下のように扱います。

- `Encoder::distance()` は距離を 0–25 に補正します。0 は可逆です。
- `Encoder::effort()` は値を 1–10 に補正します。
- `Encoder::threads()` は 0 を 1 に補正します。
- `Encoder::quality()` は品質を距離に変換します。100 以上は可逆になりますが、負の値を 0 に補正する処理はありません。品質は 0–100 の有限値を指定してください。距離にも有限値を指定してください。

`EncoderOptions` を直接渡す `encode()` でも effort とスレッド数は補正されますが、距離は補正されません。距離には 0–25 の有限値を指定してください。

## エンコード機能

- 逆 Gaborish と量子化のブロック別調整（AdjustQuantBlockAC）は effort 5 以上で有効です。逆 Gaborish は距離 0.5 超でのみ有効です。`MINI_NO_GABORISH=1` / `MINI_NO_ADJUST_QUANT=1` で無効化できます。
- パッチは effort 7 以上で有効です。テキストや UI の繰り返し図形を参照フレームに符号化します。`MINI_NO_PATCHES=1` で無効化できます。
- AFV、ノイズ合成、スプライン、butteraugli による反復量子化は未実装です。

## 構成

| モジュール | 内容 |
|---|---|
| `bit_writer` | LSB 先頭のビットライタ、U32/U64/F16 |
| `entropy` | hybrid-uint、コンテキストクラスタリング、ANS（分布の正規化とエイリアス表） |
| `headers` | SizeHeader / ImageMetadata / FrameHeader / TOC |
| `modular` | 予測器・プロパティ・MA 木学習・RCT |
| `lossless` | 可逆フレーム（グローバル木 + グループ分割） |
| `vardct` | XYB、DCT、AQ、AC 戦略探索、係数のトークン化、LF/HF メタデータ |
| `patches` | パッチの検出、参照フレーム（Modular）と辞書の符号化 |

## 検証

```sh
cargo test --release                 # jxl-oxide でラウンドトリップ（可逆は完全一致）
cargo run --release --example eval -- -q 90 -e 7 image.png   # サイズ・PSNR・SSIMULACRA2
```

ライセンスは libjxl に合わせて BSD-3-Clause です。
