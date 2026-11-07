# jxl-mini-encoder

8 bit sRGB の RGB / RGBA 入力に対応した、純 Rust の JPEG XL エンコーダです。C/C++ ツールチェーン・CMake・bindgen は不要です。

libjxl のエンコード処理・ヒューリスティクス・定数を元に実装しています。対応機能やパラメータの扱いには独自の変更があり、libjxl と同じ挙動を保証するものではありません。

対象範囲:

- 入力: 8 bit sRGB の RGB / RGBA（ストレートアルファ）
- 非可逆: VarDCT（XYB、適応量子化、DCT8〜DCT64 の可変ブロック、EPF、パッチ）
- 可逆（品質 100）: Modular（RCT 選択、MA 木学習、重み付き予測器。写真向けに調整した予測器パラメータをヘッダーで宣言する）
- effort 1–10、スレッド数指定
- 出力: 裸のコードストリーム（`FF 0A`）。アニメーション（全面フレーム）に対応。ICC・プログレッシブ・デコーダは対象外

## 使い方

```rust
let image = jxl_mini_encoder::ImageRef::rgba8(width, height, &pixels)?;
let jxl = jxl_mini_encoder::Encoder::new()
    .quality(75.0) // 100 で可逆
    .effort(7)
    .threads(4)
    .encode(&image)?;
```

## アニメーション

デコード済みの GIF などのフレーム列を、アニメーション対応 JXL にエンコードできます。各フレームは合成済みのキャンバス全体（同じサイズ・同じアルファ有無）で渡してください。非可逆・可逆は距離の指定で全フレーム共通に決まります。フレームは並列にエンコードされ（`threads` を使い切ります）、出力はスレッド数に依存しません。直前のキャンバスから変わっていない領域は符号化せず、変化した矩形だけを切り出しフレームとして前フレームに重ね（フレーム間予測）、変化のないフレームは前フレームの表示時間に加算します（`MINI_NO_INTERFRAME=1` で無効化）。背景が固定の素材ではサイズが大きく減り、画素が毎フレーム変わる素材では従来と同じ出力です。

```rust
use jxl_mini_encoder::{AnimationFrame, AnimationOptions, Encoder, ImageRef};

let frames: Vec<AnimationFrame> = decoded_gif_frames
    .iter()
    .map(|f| AnimationFrame {
        image: ImageRef::rgba8(width, height, &f.rgba).unwrap(),
        duration: f.delay_cs, // 既定は 100 tick/秒なので GIF の遅延（1/100 秒）をそのまま指定
    })
    .collect();
let jxl = Encoder::new()
    .quality(80.0)
    .encode_animation(&frames, &AnimationOptions::default())?; // num_loops: 0 = 無限ループ
```

可逆（品質 100）では、変化した画素が変化範囲の 80% 未満のフレームを、変化していない画素を透明にしたアルファブレンドの切り出しフレームとして符号化します。星のちらつきのように変化が画面全体に散らばっていても、変化した画素だけにビットを使います。この方式では入力が RGB でもアルファチャンネル付き（全面不透明）で出力されることがあります。変化が全面に及ぶフレームは従来どおり置換フレームです。

可逆ではフレームの色数が 256 以下ならパレット変換を使います（GIF 由来の素材、画面録画など）。色数が多いフレームは従来どおり RCT です。パレットの索引チャンネルには、定数・隣接画素の予測器と隣接索引での分岐を使います。

GIF 由来のようなフラットな素材は、変化が少ないと可逆のほうが非可逆より小さくなります。`animation_prefers_lossless(&frames)` は、全フレームが 256 色以下で、フレーム間で変化する画素が平均 10% 未満のときに `true` を返します。この場合は品質指定にかかわらず可逆（`Encoder::lossless()`）で符号化すると、画質が上がりサイズも小さくなる傾向があります。

PNG 連番から試すには `cargo run --release --example anim -- -q 80 -e 7 --delay 6 -o out.jxl frames/*.png` を使います（復号して SSIMULACRA2 も表示します。`--auto` を付けると `animation_prefers_lossless` が `true` のとき可逆にします）。

`duration` が 0 のフレームは次のフレームと同時に表示されるレイヤになり、デコーダは単独の表示フレームとして扱いません。GIF の遅延 0 は、ブラウザと同様に 10（1/10 秒）などへ置き換えてから渡してください。

## パラメータの扱い

範囲外の値に対する処理は libjxl と異なります。libjxl の距離・effort 設定 API が範囲外の値をエラーにするのに対し、このライブラリでは以下のように扱います。

- `Encoder::distance()` は距離を 0–25 に補正します。0 は可逆です。
- `Encoder::effort()` は値を 1–10 に補正します。
- `Encoder::threads()` は 0 を 1 に補正します。
- `Encoder::quality()` は品質を距離に変換します。100 以上は可逆になりますが、負の値を 0 に補正する処理はありません。品質は 0–100 の有限値を指定してください。距離にも有限値を指定してください。

`EncoderOptions` を直接渡す `encode()` でも effort とスレッド数は補正されますが、距離は補正されません。距離には 0–25 の有限値を指定してください。

## エンコード機能

- 逆 Gaborish と量子化のブロック別調整（AdjustQuantBlockAC）は effort 5 以上で有効です。逆 Gaborish は距離 0.5 超で、平坦な図形画像と判定されない場合に有効です（最大1024個の8×8領域を標本化し、85%以上がRGB一定なら無効化）、かつノイズが少ない場合に有効です（輝度のラプラシアン絶対値の中央値が8以上、つまりビデオ・GIFのディザ・センサノイズが強い画像では、鮮鋭化がノイズを増幅してサイズが増えるため無効化。アニメーションでは全フレームの中央値で1回だけ決めます）。無効時はAQの目標値も調整します。`MINI_NO_GABORISH=1` / `MINI_NO_ADJUST_QUANT=1` でも無効化できます。
- effort 6 以上（距離 0.5 以上）では libjxl と同様に、ブロックごとの EPF 強度をエンコーダ内でデコード（逆量子化・逆 DCT・Gaborish・EPF）して選びます。候補（距離 4.5 以下は {0, 2, 7}、超は {0, 4}）ごとに輝度マスク付き誤差を比べ、最小の強度を採用します。処理は 16 ブロック行の帯ごとに行うため、作業メモリは画像全体に比例しません。`MINI_NO_DYNAMIC_EPF=1` で無効化できます。
- effort 5 以下、または上記を無効にした場合は定数強度です。逆Gaborish使用時の距離1.5以上ではEPFの平滑化を少し弱めて写真の細部を保持します（高品質域・逆Gaborish無効時は変更していません）。
- effort 7 以上では libjxl と同様に、画素統計から X / B チャンネルの量子化スケール（x_qm_scale / b_qm_scale）を引き上げます。`MINI_NO_CHROMA_ADJUST=1` で無効化できます。
- 適応量子化の係数には、libjxl と同じ品質・effort での SSIMULACRA2 を揃えるための較正係数（`aq_calibration`）を掛けています。effort 6–7 で最大 +5%（低品質ほど大きい）、effort 8 以上で −4% です。libjxl は effort 8 以上で Butteraugli 反復量子化を使うため、出力サイズ・スコアの動作点が異なります。
- パッチは effort 7 以上で有効です。テキストや UI の繰り返し図形を参照フレームに符号化します。`MINI_NO_PATCHES=1` で無効化できます。
- ノイズ合成は `Encoder::noise(強度)` で有効にできます（既定は無効。SSIMULACRA2 は下がります）。
- AFV、スプライン、butteraugli による反復量子化は未実装です。

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

ピークメモリを比較するときは、先に `cargo build --release --example eval` を実行し、
`/usr/bin/time -v target/release/examples/eval --encode-only ...` で直接バイナリを計測してください。
`--encode-only` は復号・画質指標の計算を省き、入力読み込みとエンコーダのみのプロセスRSSを測定できます。
出力サイズ・エンコード時間・出力ハッシュも表示します。
2つのバイナリの反復比較・同サイズ画質比較には `tools/compare_photos.py` を使用できます。


libjxl の `cjxl` と同じ品質・effort で比較するには、`eval` をビルドしたうえで次を実行します（どちらも同じデコーダと SSIMULACRA2 で採点します）。

```sh
tools/compare_libjxl.py --cjxl /path/to/cjxl -q 50 -q 75 -q 90 -e 7 images/*.png
```

ライセンスは libjxl に合わせて BSD-3-Clause です。

## CI・リリース

PR・main への push・`v*` タグの push・手動実行で、Linux x64 / arm64 のリリースビルドとテストを実行します。examples とドキュメントテストも対象です。Cargo のソースパッケージ（`.crate`）を作成し、展開したパッケージのビルド・テスト後に、SHA-256 チェックサムとともに Actions の `packages` artifact に保存します。

`Cargo.toml` と `Cargo.lock` のバージョンを更新して main にマージすると、対応する `v<version>` タグが未作成の場合にタグと GitHub Release を作成し、`.crate` と `SHA256SUMS` を添付します。バージョンと一致するタグの push でもリリースできます。crates.io への公開は行いません。
