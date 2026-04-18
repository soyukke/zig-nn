# Zig NN Library

## PR前のローカルチェック

```sh
just test-diff-cpu
just test-diff-mps
just test-diff-cuda
```

## ビルド注意事項

- Nix 環境では `NIX_CFLAGS_COMPILE=""` を zig build の前に付ける（justfile で設定済み）
