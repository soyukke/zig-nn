const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf_mod = @import("../gguf/gguf.zig");

// ============================================================
// SentencePiece Tokenizer (Greedy Longest Match)
// ============================================================
//
// Gemma3 等の SentencePiece ベースモデル用。
// GGUF メタデータの tokenizer.ggml.tokens から vocab を読み、
// Greedy longest-match でエンコードする。
//
// SentencePiece のトークンは以下の形式:
// - 通常の文字列トークン ("hello", "world" など)
// - "▁" (U+2581, 下線付きスペース) = 単語先頭マーカー
// - <s>, </s>, <pad>, <unk> 等の特殊トークン
// - <0xNN> 形式のバイトフォールバックトークン

pub const SentencePieceTokenizer = struct {
    const Self = @This();

    vocab: [][]const u8, // token_id → token string
    vocab_count: usize,
    token_map: std.StringHashMap(u32), // token string → token_id
    bos_id: u32,
    eos_id: u32,
    unk_id: u32,
    add_bos: bool,
    allocator: Allocator,

    /// GGUF メタデータから SentencePiece トークナイザーを構築
    pub fn initFromGGUF(gguf_file: *const gguf_mod.GGUFFile, allocator: Allocator) !Self {
        const vocab_strings = try gguf_file.getArrayStrings("tokenizer.ggml.tokens", allocator);
        defer allocator.free(vocab_strings);

        const vocab_count = vocab_strings.len;

        const vocab = try allocator.alloc([]const u8, vocab_count);
        for (0..vocab_count) |i| {
            const copy = try allocator.alloc(u8, vocab_strings[i].len);
            @memcpy(copy, vocab_strings[i]);
            vocab[i] = copy;
        }

        var token_map = std.StringHashMap(u32).init(allocator);
        for (0..vocab_count) |i| {
            try token_map.put(vocab[i], @intCast(i));
        }

        // Special token IDs
        const bos_id: u32 = if (gguf_file.getMetadataU32("tokenizer.ggml.bos_token_id")) |v|
            v
        else
            2;
        const eos_id: u32 = if (gguf_file.getMetadataU32("tokenizer.ggml.eos_token_id")) |v|
            v
        else
            1;
        const unk_id: u32 = if (gguf_file.getMetadataU32("tokenizer.ggml.unknown_token_id")) |v|
            v
        else
            3;

        // Check add_bos metadata
        var add_bos = true;
        if (gguf_file.getMetadata("tokenizer.ggml.add_bos_token")) |val| {
            switch (val) {
                .bool_ => |b| add_bos = b,
                else => {},
            }
        }

        return .{
            .vocab = vocab,
            .vocab_count = vocab_count,
            .token_map = token_map,
            .bos_id = bos_id,
            .eos_id = eos_id,
            .unk_id = unk_id,
            .add_bos = add_bos,
            .allocator = allocator,
        };
    }

    // SentencePiece: 先頭にスペースを追加
    // "▁" (U+2581) = 0xE2 0x96 0x81 in UTF-8
    const SP_PREFIX = "\xe2\x96\x81";

    const Match = struct { len: usize, id: u32 };

    /// 先頭に "▁" を付与した最長一致を試みる
    fn matchWithSpPrefix(self: *const Self, text: []const u8, pos: usize) Match {
        var best: Match = .{ .len = 0, .id = self.unk_id };
        var try_len: usize = @min(text.len - pos, 64);
        while (try_len > 0) : (try_len -= 1) {
            const candidate_len = SP_PREFIX.len + try_len;
            if (candidate_len > 128) continue;
            var buf: [128]u8 = undefined;
            @memcpy(buf[0..SP_PREFIX.len], SP_PREFIX);
            @memcpy(buf[SP_PREFIX.len..][0..try_len], text[pos..][0..try_len]);

            if (self.token_map.get(buf[0..candidate_len])) |id| {
                if (try_len > best.len) best = .{ .len = try_len, .id = id };
                break;
            }
        }
        return best;
    }

    /// prefix 無しの最長一致を試みる (現状の best を更新する形で返す)
    fn matchPlain(self: *const Self, text: []const u8, pos: usize, current: Match) Match {
        var best = current;
        var try_len: usize = @min(text.len - pos, 64);
        while (try_len > 0) : (try_len -= 1) {
            if (self.token_map.get(text[pos..][0..try_len])) |id| {
                if (try_len > best.len) best = .{ .len = try_len, .id = id };
                break;
            }
        }
        return best;
    }

    /// 1 バイトのフォールバック (<0xNN> または unk) をトークン列に追記
    fn appendByteFallback(
        self: *const Self,
        byte: u8,
        tokens: *std.ArrayListAligned(u32, null),
        allocator: Allocator,
    ) !void {
        var hex_buf: [6]u8 = undefined;
        const hex_str = std.fmt.bufPrint(
            &hex_buf,
            "<0x{X:0>2}>",
            .{byte},
        ) catch unreachable;
        if (self.token_map.get(hex_str)) |id| {
            try tokens.append(allocator, id);
        } else {
            try tokens.append(allocator, self.unk_id);
        }
    }

    /// テキストをトークン列にエンコード (Greedy longest match)
    pub fn encode(self: *const Self, text: []const u8, allocator: Allocator) ![]u32 {
        var tokens: std.ArrayListAligned(u32, null) = .empty;
        defer tokens.deinit(allocator);

        // Add BOS if configured
        if (self.add_bos) {
            try tokens.append(allocator, self.bos_id);
        }

        var pos: usize = 0;
        var is_word_start = true;

        while (pos < text.len) {
            var best: Match = .{ .len = 0, .id = self.unk_id };
            if (is_word_start) best = self.matchWithSpPrefix(text, pos);
            best = self.matchPlain(text, pos, best);

            if (best.len == 0) {
                try self.appendByteFallback(text[pos], &tokens, allocator);
                pos += 1;
                is_word_start = false;
            } else {
                try tokens.append(allocator, best.id);
                pos += best.len;
                // After a space, next token is word start
                is_word_start = (pos > 0 and text[pos - 1] == ' ');
            }
        }

        return try tokens.toOwnedSlice(allocator);
    }

    /// トークン列をテキストにデコード
    pub fn decode(self: *const Self, tokens: []const u32, allocator: Allocator) ![]u8 {
        var result: std.ArrayListAligned(u8, null) = .empty;

        for (tokens) |tid| {
            if (tid == self.bos_id or tid == self.eos_id) continue;
            if (tid < self.vocab_count) {
                const tok = self.vocab[tid];
                // "▁" → " " (space)
                var i: usize = 0;
                while (i < tok.len) {
                    if (i + 2 < tok.len and
                        tok[i] == 0xE2 and tok[i + 1] == 0x96 and tok[i + 2] == 0x81)
                    {
                        try result.append(allocator, ' ');
                        i += 3;
                    } else if (tok.len >= 5 and tok[0] == '<' and tok[1] == '0' and tok[2] == 'x') {
                        // <0xNN> byte fallback
                        const byte = std.fmt.parseInt(u8, tok[3..5], 16) catch {
                            try result.append(allocator, tok[i]);
                            i += 1;
                            continue;
                        };
                        try result.append(allocator, byte);
                        break;
                    } else {
                        try result.append(allocator, tok[i]);
                        i += 1;
                    }
                }
            }
        }

        return try result.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *Self) void {
        for (self.vocab) |v| self.allocator.free(v);
        self.allocator.free(self.vocab);
        self.token_map.deinit();
    }
};
