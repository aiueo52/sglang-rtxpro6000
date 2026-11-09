# Expert tier

Expert tier は、MoE の専門家の重みを置く、任意で有効にする LRU の置き場です。
各層の K 人を VRAM に置き、不足する専門家を host の mmap ソースからコピーします。
hot set は初期配置を決めるもので、その専門家を固定し続ける仕組みではありません。
route の要素数が K 以下の呼び出しは decide/gather、それより大きい呼び出しは
arena を使います。K を減らすと転送量が増え、速度が下がることがあります。

## 使い方

既定は切です。`EXPERT_TIER_K=0` では置き場用の環境変数を追加せず、既存の
起動動作と引き継いだ SGLang の環境設定を保ちます。有効にするにはリポジトリの
ルートで起動します。公開の文から作った hot set を `expert-tier/hot-set.json`
に同梱しています（「hot set の作成」を参照）。

```bash
TARGET_MODEL=models/model CONC=8 EXPERT_TIER_K=320 ./serve-fast.sh wa
```

自分の負荷に合わせるには、先に trace から hot set を作り直します。

```bash
python tools/build_hot_set.py traces/trace-000000.npz traces/trace-000001.npz \
  --output expert-tier/hot-set.json
```

この例は既定の context 長 262,144 を使います。K > 0 では、`MAX_TOTAL_TOKENS` が
未設定または空なら `CONTEXT_LENGTH` の値（未設定または空なら 262,144）にして
export します。利用者が指定した空でない `MAX_TOTAL_TOKENS` はそのまま export
します。この置き場用の既定設定で `CONTEXT_LENGTH` 自体は変更しません。
K=0 では既存のトークン予算の動作を変えません。

VRAM 使用量を減らすなら `MAX_TOTAL_TOKENS` を下げ、空いた VRAM をより多く
KV cache に回すなら上げます。他の設定と同様、起動時の環境変数で指定します。
`WA_MEM_FRACTION`（既定 0.96）もメモリ予算を制御し、起動スクリプトは有効な
割合から画面用の予約分を差し引きます。

`CONTEXT_LENGTH` が 262144 を越えると、`serve-local.sh` が
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` を設定し、
`--json-model-override-args` で `text_config.rope_parameters` を追加します。
内容は `rope_type` yarn、`factor` 2.0、`original_max_position_embeddings` 262144、
`mrope_interleaved` true、`mrope_section` [11,11,10]、`rope_theta` 10000000、
`partial_rotary_factor` 0.25 です。モデル設定を書き換える必要はありません。

| 設定 | 既定 | 意味 |
| --- | --- | --- |
| `EXPERT_TIER_K` | `0` | 各層の常駐人数。整数 0..512。0 で起動スクリプトの置き場設定を無効化 |
| `EXPERT_TIER_HOT` | リポジトリ内の `expert-tier/hot-set.json` | 初期配置 JSON。K > 0 ではファイルが必要 |
| `EXPERT_TIER_KEEP` | `60` | host ページの keeper の間隔、秒単位。0 は無効、それ以外は 5..3600 |
| `ADAPTIVE_CONFIG` | 既存の起動スクリプトの選択 | 警告計算と起動で実際に使う adaptive 表 |

既存の表選択は維持します。CONC=8 の既定は `adaptive/w16_c8.json`、
それ以外の対応する同時実行数では `adaptive/w16_c4.json` です。
別の表は `ADAPTIVE_CONFIG` で指定できます。CONC=6 用の
`adaptive/w16_c6.json` を使う場合も明示してください。

K > 0 では測定時の設定を export します。prologue 外の刈り込み
(`SGLANG_MOE_PRUNE_IN_PROLOGUE=0`)、source mmap、gather v2、chunk 8192、
CTAs 376、pretouch 1、decide sort、predict 0、timing 0、arena v2、prefetch 0、
fused 0、policy lru、`SGLANG_RND_PLE_META=1` です。K・HOT・KEEP は
対応する `SGLANG_EXPERT_TIER_*` に渡します。

同じ環境で `./serve-fast.sh wa --print` を実行すると、サーバーを起動せず、
キャッシュ用ディレクトリも作らずに、解決した環境変数と引数を JSON で表示します。
この場合も hot set がなければエラーになります。

警告で用いる M は、起動で対応する各バッチ数を、その数以下で最大の表の bucket
（なければ最初の bucket）へ振り分け、候補の `batch_size * (candidate_steps + 1)`
の最大値を取ります。追加の 1 行は verify/extend 用です。top-k が 10 なので、
速い経路の条件は `10 * M <= K` です。c8 表は M=32、必要な K=320 です。
不足時は stderr に警告を出して続行します。この計算は adaptive の検証に対する
もので、prefill や明示的な起動引数の上書きでは大きな呼び出しになり、arena を
使うことがあります。

## hot set の作成

代表的な負荷で置き場を有効にして実行し、`SGLANG_EXPERT_TIER_TRACE` に
trace ディレクトリの絶対パスを指定すると、`trace-*.npz` を記録します。
記録には負荷があるので、通常の serving では指定しません。必要なファイルを
`build_hot_set.py` に渡します。変換に必要なのは NumPy で、モデル・GPU・
serving モジュールの読み込みは不要です。

recorder は予測や demand の情報に加え、`hdr [N,4]` と `counts [N,512]` を
保存します。counts はトークンの route の本数を保ち、ID=-1 と重み0を個別に
除外します。ID を残した重み0の route も除外します。頻度の記録は trace を
有効にしたときだけ行い、recorder が観測する小さな decide/gather の呼び出しが
対象です。arena や prefill の呼び出しは含みません。

counts のない従来の記録は `dem` の使用有無を使うため、使用回数は「使った
呼び出しの回数」となります。この形式では route の重複本数や、ID を残して
重みだけ0にした route を復元できません。従来形式では通常の ID を除去する
刈り込みを使うか、正確な非ゼロ route の本数が必要なら再記録してください。
生の route を持つ `layers [N]`、`ids [N,...]`、同じ形の `weights` も受け付け、
ID=-1 と重み0を個別に除外します。

統一した頻度が必要な場合、従来の呼び出し回数と route 本数を混ぜないでください。
複数ファイルは合算し、同じファイルパスの重複指定は拒否します。ring の上書きで
失った記録は復元できません。別の実行で採取した記録の重複にも注意してください。

出力は `{"K": 512, "layers": [...]}` で、48層それぞれに全512人の番号を並べます。
使用回数の降順、同数では番号の昇順とし、未使用の専門家も含めます。
実行時は各層の先頭 K 人を読みます。初期配置の効果は負荷に依存します。

同梱の `expert-tier/hot-set.json` は、K320・`CONC=8` の実行で公開の文だけを
流して記録した 109 個の trace から、この手順で作りました。流した文は、
OpenAssistant oasst2 の日本語の会話 64 件の最初の利用者の発話、GSM8K の
test の問い 256 問（思考なし）、CPython 3.13.0 の `Lib/` のソースを使う混線の
確認です。初期配置が変えるのは転送の量だけで、各トークンがどの専門家に
振り分けられるかは変わりません。

## 前提と限り

対応するのは ModelOpt NVFP4 MoE、FlashInfer CUTLASS backend、MoE TP=1・EP=1、
48層・512人・top-k 10 の構成です。interleaved w13、fused shared experts、
対象層の draft MoE GEMV、動作中の OEA prologue には対応しません。
互換性のない prologue を有効にせず、normal edition の構成を使ってください。
CPU 試験はロジックの検証であり、GPU serving の互換性の確認ではありません。

mmap ソースとページ常駐を支える host RAM とストレージが必要です。VRAM を
減らすと host 側のメモリと転送の負担が増えます。K は層ごとの重みの常駐予算で、
VRAM 全体の上限ではありません。KV cache、context、同時実行数、arena なども
VRAM の山に影響します。転送や draft の採択率でも速度は変わり、以下の値は
特定構成の測定結果です。モデルの重みは別途用意します。

H40（VT MoE prologue 向けの hyperburst FlashInfer ビルド時パッチ、
`patches/flashinfer/2026-11-08`）と expert tier の併用は未検証です。
`flashinfer-vt` にこのパッチがあると、`serve-fast.sh` は
`EXPERT_TIER_K>0` を拒否します。既存の `FLASHINFER_WORKSPACE_BASE` にある、
H40 ソースからコンパイル済みのモジュールは検出できません。
パッチを外した後、新しい workspace でビルドしてください。

## 測った値

以下は提供された測定結果で、公開準備の CPU 試験で再測定した値ではありません。

* この launcher をそのまま使った構成。`CONC=8 EXPERT_TIER_K=320 ./serve-fast.sh wa`
  と同梱の hot set、context 262,144・`MAX_TOTAL_TOKENS=262144`・HiCache 32 GB
  （既定）・prefill chunk 8192・実効の mem fraction 0.886778:
  VRAM の山 **66,230 MiB**。実行中に host の MemAvailable は約 **107 GB**
  減りました（HiCache の 32 GB を含みます）。

| 同時実行数 | 1本あたり tok/s | 全体 tok/s |
| --- | ---: | ---: |
| c1 | 232 | 232 |
| c2 | 206 | 409 |
| c4 | 157 | 610 |
| c8 | 112 | 869 |

置き場 K320・context 262,144 の profile W4（draft 4・adaptive なし、以前の
起動方法）と比べ、1本あたりの速さは c1 と c2 で差 2% 以内、c4 で 2.4% 遅く、
c8 で 10% 速い結果でした（全体 869 対 791）。コード編集の c1 は **615 tok/s**、
採択の長さの平均は約 10 でした。日本語の 8 件の入力（1,479 トークン）の
logprob はその比較元と完全に同じでした。GSM8K 1,319問は **545 対 570** で、
比較元だけが正解した問いが 121、この起動だけが正解した問いが 96 でした
（exact McNemar p ≈ 0.10）。下の 524k の 2 つの構成も、同じ比較元に対して
549（W4）と 552（w16_c8）でした。63k の混線は **8/8・漏れ0** でした。

* K320・context 524,288（YaRN factor 2）・profile wa（表 w16_c8）:
  VRAM の山 **72,922 MiB**。524k の needle **3/3**、最初の1文字まで **78.6秒**。
  この測定は以前の起動方法で、running 8・graph bs 1〜8・mamba 40・prefill chunk
  16384・`MAX_TOTAL_TOKENS=557056`・mem fraction 0.886778・HiCache なしの構成です。
  この launcher の現在の `CONC=8` の経路で測った値ではありません。

| 同時実行数 | 1本あたり tok/s | 全体 tok/s |
| --- | ---: | ---: |
| c1 | 227 | 227 |
| c2 | 206 | 402 |
| c4 | 158 | 623 |
| c8 | 97 | 763 |

同じ置き場の構成（K320・context 524,288）で、profile W4（draft 4・adaptive なし）
と比べて質の差は観測されていません。
GSM8K 1,319問は **552 対 549**、思考あり200問 × 3回は **585 対 586**、
63k の混線は **8/8・漏れ0** でした。日本語の文の c1 は、その同じ profile W4
（draft 4）の構成より **約7%遅い**結果でした。長い draft が当たりにくいためです。
質に関する比較の範囲は、これらの負荷での確認に限ります。

* 省 VRAM 構成 K216・context 262,144・profile W4（draft 4）・running 4・
  `MAX_TOTAL_TOKENS=262144`:
  VRAM の山 **48,172 MiB**、速さは置き場なしの **86%**（**5回の測定の平均**）。
