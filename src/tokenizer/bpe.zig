const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf_mod = @import("../gguf/gguf.zig");

// ============================================================
// GPT-2 Byte-level BPE Tokenizer
// ============================================================

pub const BPETokenizer = struct {
    const Self = @This();

    vocab: [][]const u8, // token_id → token string
    vocab_count: usize,
    token_map: std.StringHashMap(u32), // token string → token_id
    merges: []MergePair,
    merge_map: std.StringHashMap(u32), // "first second" → merge priority
    allocator: Allocator,

    pub const MergePair = struct {
        first: []const u8,
        second: []const u8,
    };

    /// GGUF メタデータから BPE トークナイザーを構築
    pub fn init_from_gguf(gguf_file: *const gguf_mod.GGUFFile, allocator: Allocator) !Self {
        // Vocab
        const vocab_strings = try gguf_file.get_array_strings("tokenizer.ggml.tokens", allocator);
        defer allocator.free(vocab_strings);

        const vocab_count = vocab_strings.len;

        // vocab をコピー
        const vocab = try allocator.alloc([]const u8, vocab_count);
        for (0..vocab_count) |i| {
            const copy = try allocator.alloc(u8, vocab_strings[i].len);
            @memcpy(copy, vocab_strings[i]);
            vocab[i] = copy;
        }

        // token → id マップ
        var token_map = std.StringHashMap(u32).init(allocator);
        for (0..vocab_count) |i| {
            try token_map.put(vocab[i], @intCast(i));
        }

        // Merges
        const merges_raw = try gguf_file.get_array_strings("tokenizer.ggml.merges", allocator);
        defer allocator.free(merges_raw);

        const merges = try allocator.alloc(MergePair, merges_raw.len);
        var merge_map = std.StringHashMap(u32).init(allocator);

        for (0..merges_raw.len) |i| {
            const line = merges_raw[i];
            // フォーマット: "first second"
            const space_pos = std.mem.indexOf(u8, line, " ") orelse continue;
            const first = try allocator.dupe(u8, line[0..space_pos]);
            const second = try allocator.dupe(u8, line[space_pos + 1 ..]);
            merges[i] = .{ .first = first, .second = second };

            // merge key を構築
            const merge_key = try allocator.alloc(u8, line.len);
            @memcpy(merge_key, line);
            try merge_map.put(merge_key, @intCast(i));
        }

        return .{
            .vocab = vocab,
            .vocab_count = vocab_count,
            .token_map = token_map,
            .merges = merges,
            .merge_map = merge_map,
            .allocator = allocator,
        };
    }

    /// テキストをトークン列にエンコード (Byte-level BPE)
    pub fn encode(self: *const Self, text: []const u8, allocator: Allocator) ![]u32 {
        if (text.len == 0) {
            return try allocator.alloc(u32, 0);
        }

        // Step 1: テキストを UTF-8 バイトトークン列に分解
        var tokens: std.ArrayListAligned([]const u8, null) = .empty;
        defer tokens.deinit(allocator);

        for (text) |byte| {
            const byte_str = byte_to_token(byte);
            try tokens.append(allocator, byte_str);
        }

        // Step 2: BPE マージを繰り返し適用
        while (tokens.items.len > 1) {
            // 最も優先度の高いマージペアを見つける
            var best_idx: ?usize = null;
            var best_priority: u32 = std.math.maxInt(u32);

            for (0..tokens.items.len - 1) |i| {
                const pair_key = merge_pair_key(
                    tokens.items[i],
                    tokens.items[i + 1],
                    allocator,
                ) catch continue;
                defer allocator.free(pair_key);

                if (self.merge_map.get(pair_key)) |priority| {
                    if (priority < best_priority) {
                        best_priority = priority;
                        best_idx = i;
                    }
                }
            }

            if (best_idx == null) break;

            // マージ実行: tokens[best_idx] と tokens[best_idx+1] を結合
            const idx = best_idx.?;
            const merged = try std.mem.concat(
                allocator,
                u8,
                &[_][]const u8{ tokens.items[idx], tokens.items[idx + 1] },
            );
            tokens.items[idx] = merged;
            _ = tokens.orderedRemove(idx + 1);
        }

        // Step 3: 文字列トークンを ID に変換
        const ids = try allocator.alloc(u32, tokens.items.len);
        for (tokens.items, 0..) |tok, i| {
            ids[i] = self.token_map.get(tok) orelse 0; // fallback to 0
        }

        return ids;
    }

    /// トークン列をテキストにデコード
    pub fn decode(self: *const Self, tokens: []const u32, allocator: Allocator) ![]u8 {
        var result: std.ArrayListAligned(u8, null) = .empty;
        for (tokens) |tid| {
            if (tid < self.vocab_count) {
                const tok = self.vocab[tid];
                var i: usize = 0;
                while (i < tok.len) {
                    const byte = token_char_to_byte(tok, &i);
                    try result.append(allocator, byte);
                }
            }
        }
        return try result.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *Self) void {
        for (self.vocab) |v| self.allocator.free(v);
        self.allocator.free(self.vocab);
        for (self.merges) |m| {
            self.allocator.free(m.first);
            self.allocator.free(m.second);
        }
        self.allocator.free(self.merges);

        var merge_it = self.merge_map.iterator();
        while (merge_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.merge_map.deinit();
        self.token_map.deinit();
    }
};

// ============================================================
// GPT-2 バイト ↔ Unicode コードポイント変換
// ============================================================
//
// GPT-2 byte_encoder マッピング:
//   直接: bytes {33-126} ∪ {161-172} ∪ {174-255} → 同じ Unicode コードポイント
//   間接: 残り68バイト (0-32, 127-160, 173) → U+0100 + n (順番にマッピング)
//
// BPE トークン文字列は UTF-8 エンコードされたコードポイント列。

/// バイト → GPT-2 BPE トークン文字列 (UTF-8 encoded)
fn byte_to_token(byte: u8) []const u8 {
    return BYTE_TO_TOKEN_TABLE[byte].slice();
}

/// GPT-2 BPE トークン文字列から 1 文字分をデコードして元のバイトを返す
fn token_char_to_byte(tok: []const u8, pos: *usize) u8 {
    if (pos.* >= tok.len) return 0;

    const b0 = tok[pos.*];

    // 1-byte UTF-8 (ASCII: U+0000-U+007F)
    if (b0 < 0x80) {
        pos.* += 1;
        return CP_TO_BYTE_TABLE[b0];
    }

    // 2-byte UTF-8 (U+0080-U+07FF)
    if (pos.* + 1 < tok.len and (b0 & 0xE0) == 0xC0) {
        const b1 = tok[pos.* + 1];
        if ((b1 & 0xC0) == 0x80) {
            pos.* += 2;
            const cp: u16 = (@as(u16, b0 & 0x1F) << 6) | @as(u16, b1 & 0x3F);
            return CP_TO_BYTE_TABLE[cp];
        }
    }

    // Fallback
    pos.* += 1;
    return b0;
}

fn is_direct_byte(b: u8) bool {
    return (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174);
}

/// バイト → UTF-8 トークン文字列テーブル (comptime)
const TokenEntry = struct {
    data: [3]u8,
    len: u8,

    fn slice(self: *const TokenEntry) []const u8 {
        return self.data[0..self.len];
    }
};

const BYTE_TO_TOKEN_TABLE: [256]TokenEntry = init_byte_table();

fn init_byte_table() [256]TokenEntry {
    var table: [256]TokenEntry = undefined;
    var n: u16 = 0;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (is_direct_byte(b)) {
            if (b < 0x80) {
                table[i] = .{ .data = .{ b, 0, 0 }, .len = 1 };
            } else {
                // Latin-1 → UTF-8 (2 bytes)
                table[i] = .{ .data = .{ 0xC0 | (b >> 6), 0x80 | (b & 0x3F), 0 }, .len = 2 };
            }
        } else {
            // U+0100 + n → UTF-8 (2 bytes)
            const cp: u16 = 256 + n;
            table[i] = .{
                .data = .{ @intCast(0xC0 | (cp >> 6)), @intCast(0x80 | (cp & 0x3F)), 0 },
                .len = 2,
            };
            n += 1;
        }
    }
    return table;
}

/// Unicode コードポイント → 元のバイト逆変換テーブル (comptime)
/// 最大コードポイントは U+0100+67 = U+0143 = 323
const CP_TO_BYTE_TABLE: [324]u8 = init_cp_to_byte();

fn init_cp_to_byte() [324]u8 {
    @setEvalBranchQuota(5000);
    var table: [324]u8 = undefined;
    @memset(&table, 0);

    // 直接マッピング: byte == codepoint
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (is_direct_byte(b)) {
            table[i] = b;
        }
    }

    // 間接マッピング: U+0100 + n → 元のバイト
    var n: u16 = 0;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (!is_direct_byte(b)) {
            table[256 + n] = b;
            n += 1;
        }
    }

    return table;
}

/// マージペアのキー文字列を構築: "first second"
fn merge_pair_key(first: []const u8, second: []const u8, allocator: Allocator) ![]u8 {
    const key = try allocator.alloc(u8, first.len + 1 + second.len);
    @memcpy(key[0..first.len], first);
    key[first.len] = ' ';
    @memcpy(key[first.len + 1 ..], second);
    return key;
}

// ============================================================
// テスト
// ============================================================

test "byteToToken ASCII printable" {
    // 'A' (65) is in printable range → maps to itself
    const tok = byte_to_token('A');
    try std.testing.expectEqual(@as(u8, 'A'), tok[0]);
}

test "byteToToken space" {
    // space (32) is NOT in the direct range → maps to U+0100+n
    const tok = byte_to_token(' ');
    // Should be a 2-byte UTF-8 sequence (U+0120 = 0xC4 0xA0)
    try std.testing.expect(tok[0] >= 0xC0);
}

test "tokenCharToByte roundtrip" {
    // Test that byteToToken → tokenCharToByte roundtrips for all 256 bytes
    for (0..256) |i| {
        const byte: u8 = @intCast(i);
        const tok = byte_to_token(byte);
        var pos: usize = 0;
        const decoded = token_char_to_byte(tok, &pos);
        try std.testing.expectEqual(byte, decoded);
    }
}
