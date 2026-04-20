# Logging / Profiling

`src/log.zig` が提供する scoped logger と sink の仕様。実装の正本はコード、
本ドキュメントは利用者向けのリファレンスです。

## Sink 設計

用途ごとに 3 系統に分離しています。

| 用途             | 出力先                     | 制御                        |
|------------------|----------------------------|-----------------------------|
| 通常ログ         | **stderr** (`nnLogFn`)     | `NN_LOG_LEVEL`, `NN_LOG_SCOPES` |
| モデル生成テキスト | **stdout** (call site 直)  | (なし; 通常のプログラム出力) |
| Profile dump     | **ファイル** (`ProfileArtifact`) | `NN_PROFILE`, `NN_PROFILE_DIR` |

stderr と stdout を混ぜないことで `./run > out.txt` のリダイレクトが
生成結果だけを拾えます。

## 有効化

実行ファイル (example / test root) の root module で次の 1 行を追加:

```zig
pub const std_options = @import("nn").log.std_options;
```

テストで level を絞りたい場合:

```zig
pub const std_options = @import("nn").log.stdOptionsAtLevel(.warn);
```

## ログレベル: 2 重 gate

| Gate     | 何を決めるか                     | 設定                              |
|----------|----------------------------------|-----------------------------------|
| comptime | 文字列が binary に残るか         | `std_options.log_level` (既定 `.debug`) |
| runtime  | 実行時に表示する閾値             | `NN_LOG_LEVEL` (既定 `.info`)     |

comptime を `.debug` に張っておくことで、Release build でも
`NN_LOG_LEVEL=debug` で後から詳細ログを有効化できます。

## 環境変数

### `NN_LOG_LEVEL`

- 値: `err` | `error` | `warn` | `warning` | `info` | `debug` (大小無視)
- 既定: `info`
- 例: `NN_LOG_LEVEL=debug ./a.out`

### `NN_LOG_SCOPES`

- 値: scope 名のカンマ区切り (allowlist) / `*` or `all` (全通過)
- 既定: 未指定 = 全通過
- 登録済み scope: `nn`, `cpu`, `metal`, `cuda`, `gguf`, `gemma3`,
  `trainer`, `example`, `gradcheck`
- 上限: 16 scope / 1 scope 名 32 文字まで
- 例: `NN_LOG_SCOPES=metal,cuda ./a.out`

### `NN_PROFILE`

- 値: 立っていれば有効。`0` / `false` (大小無視) / 空文字列は無効扱い
- 既定: 無効
- 作用: `openProfileArtifact()` が `null` ではなくファイルを返す
- 例: `NN_PROFILE=1 ./a.out`

### `NN_PROFILE_DIR`

- 値: profile artifact 出力ディレクトリ
- 既定: `zig-out/profiles`
- 生成ファイル名: `<prefix>-<unix_ts>.txt`
- 例: `NN_PROFILE=1 NN_PROFILE_DIR=/tmp/prof ./a.out`

## 出力フォーマット

```
[<level>][<scope>] <message>
```

`level` は `err ` / `warn` / `info` / `dbg ` の 4 文字固定幅。

## Scope の借り方

feature 単位で borrow する:

```zig
const log = @import("nn").log.metal;
log.info("using device: {s}", .{device_name});
```

新しい scope を増やす場合は `src/log.zig` に `pub const foo = std.log.scoped(.foo);`
を追加し、本ドキュメントの一覧も更新してください。

## Profile dump の使い方

```zig
var prof = (try nn.log.openProfileArtifact("gemma3")) orelse return;
defer prof.close();
try prof.writer().print("step={d} loss={d:.4}\n", .{ step, loss });
```

`NN_PROFILE` が立っていない場合は `null` が返るので、呼び出し側は
`orelse return` で no-op に畳める設計です。

## 組み合わせ例

```bash
# Metal / CUDA scope だけ debug で見る
NN_LOG_LEVEL=debug NN_LOG_SCOPES=metal,cuda zig build gemma3

# Profile artifact を /tmp に出しながら警告以上だけ表示
NN_LOG_LEVEL=warn NN_PROFILE=1 NN_PROFILE_DIR=/tmp/prof ./zig-out/bin/gemma3
```
