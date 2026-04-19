const std = @import("std");
const Allocator = std.mem.Allocator;
const dequant = @import("dequant.zig");

pub const GGMLType = dequant.GGMLType;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

/// メタデータ値の型
pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

/// 配列データ（遅延パース用）
pub const ArrayValue = struct {
    elem_type: ValueType,
    count: u64,
    raw_data: []const u8,
};

/// メタデータの値
pub const MetadataValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    array: ArrayValue,
    uint64: u64,
    int64: i64,
    float64: f64,
};

/// メタデータ KV ペア
pub const MetadataKV = struct {
    key: []const u8,
    value: MetadataValue,
};

/// テンソル情報
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dimensions: [4]u64, // 最大 4 次元、GGUF順（逆順）
    type_: GGMLType,
    offset: u64,

    /// 要素数
    pub fn numElements(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (0..self.n_dims) |i| {
            n *= self.dimensions[i];
        }
        return n;
    }
};

/// パース済み GGUF ファイル
pub const GGUFFile = struct {
    version: u32,
    metadata: []MetadataKV,
    tensors: []TensorInfo,
    data_offset: u64,
    alignment: u64,
    file_path: []const u8,
    file_buf: []const u8, // ファイル全体のバッファ（mmapの代わり）
    allocator: Allocator,

    pub fn deinit(self: *GGUFFile) void {
        self.allocator.free(self.file_buf);
        self.allocator.free(self.metadata);
        self.allocator.free(self.tensors);
        self.allocator.free(self.file_path);
    }

    /// メタデータを key で検索
    pub fn getMetadata(self: *const GGUFFile, key: []const u8) ?MetadataValue {
        for (self.metadata) |kv| {
            if (std.mem.eql(u8, kv.key, key)) return kv.value;
        }
        return null;
    }

    /// 文字列メタデータ取得
    pub fn getMetadataString(self: *const GGUFFile, key: []const u8) ?[]const u8 {
        const val = self.getMetadata(key) orelse return null;
        return switch (val) {
            .string => |s| s,
            else => null,
        };
    }

    /// u32 メタデータ取得
    pub fn getMetadataU32(self: *const GGUFFile, key: []const u8) ?u32 {
        const val = self.getMetadata(key) orelse return null;
        return switch (val) {
            .uint32 => |v| v,
            else => null,
        };
    }

    /// u64 メタデータ取得
    pub fn getMetadataU64(self: *const GGUFFile, key: []const u8) ?u64 {
        const val = self.getMetadata(key) orelse return null;
        return switch (val) {
            .uint64 => |v| v,
            else => null,
        };
    }

    /// f32 メタデータ取得
    pub fn getMetadataF32(self: *const GGUFFile, key: []const u8) ?f32 {
        const val = self.getMetadata(key) orelse return null;
        return switch (val) {
            .float32 => |v| v,
            else => null,
        };
    }

    /// テンソルを名前で検索
    pub fn getTensorInfo(self: *const GGUFFile, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    /// テンソルの生バイト列への参照を返す（コピーなし）
    pub const RawTensorRef = struct {
        data: []const u8,
        type_: GGMLType,
        n_elem: usize,
        out_dim: usize, // 2D: dimensions[1], 1D: dimensions[0]
        in_dim: usize, // 2D: dimensions[0], 1D: 1
    };

    pub fn getTensorRawBytes(self: *const GGUFFile, name: []const u8) !RawTensorRef {
        const info = self.getTensorInfo(name) orelse return error.TensorNotFound;
        const n_elem: usize = @intCast(info.numElements());
        const data_bytes = dequant.tensorBytes(info.type_, n_elem);
        const file_offset: usize = @intCast(self.data_offset + info.offset);

        if (file_offset + data_bytes > self.file_buf.len) return error.UnexpectedEof;

        const out_dim: usize = if (info.n_dims >= 2) @intCast(info.dimensions[1]) else @intCast(info.dimensions[0]);
        const in_dim: usize = if (info.n_dims >= 2) @intCast(info.dimensions[0]) else 1;

        return .{
            .data = self.file_buf[file_offset..][0..data_bytes],
            .type_ = info.type_,
            .n_elem = n_elem,
            .out_dim = out_dim,
            .in_dim = in_dim,
        };
    }

    /// テンソルデータを f32 として読み込み（逆量子化つき）
    pub fn loadTensorF32(self: *const GGUFFile, name: []const u8, allocator: Allocator) ![]f32 {
        const info = self.getTensorInfo(name) orelse return error.TensorNotFound;
        const n_elem: usize = @intCast(info.numElements());
        const data_bytes = dequant.tensorBytes(info.type_, n_elem);
        const file_offset: usize = @intCast(self.data_offset + info.offset);

        if (file_offset + data_bytes > self.file_buf.len) return error.UnexpectedEof;
        const src = self.file_buf[file_offset..][0..data_bytes];

        const dst = try allocator.alloc(f32, n_elem);
        errdefer allocator.free(dst);
        try dequant.dequantize(info.type_, src, dst, n_elem);
        return dst;
    }

    /// テンソルデータを f32 として読み込み + 2D 転置
    /// GGUF の 2D テンソルは (cols, rows) 順なので (rows, cols) に転置
    pub fn loadTensorF32Transposed(self: *const GGUFFile, name: []const u8, allocator: Allocator) ![]f32 {
        const info = self.getTensorInfo(name) orelse return error.TensorNotFound;
        if (info.n_dims != 2) return error.InvalidDimensions;

        const cols: usize = @intCast(info.dimensions[0]);
        const rows: usize = @intCast(info.dimensions[1]);
        const n_elem = rows * cols;

        // まず元データを読み込み
        const raw = try self.loadTensorF32(name, allocator);
        defer allocator.free(raw);

        // 転置: raw は (rows, cols) 格納だが GGUF順序により実質 (cols, rows) row-major
        // → 出力は (rows, cols) row-major
        const dst = try allocator.alloc(f32, n_elem);
        for (0..rows) |r| {
            for (0..cols) |c| {
                dst[r * cols + c] = raw[c * rows + r];
            }
        }
        return dst;
    }

    /// 配列メタデータの文字列要素を取得
    pub fn getArrayStrings(self: *const GGUFFile, key: []const u8, allocator: Allocator) ![][]const u8 {
        const val = self.getMetadata(key) orelse return error.KeyNotFound;
        const arr = switch (val) {
            .array => |a| a,
            else => return error.NotArray,
        };
        if (@as(u32, @intFromEnum(arr.elem_type)) != @as(u32, @intFromEnum(ValueType.string)))
            return error.NotStringArray;

        const count: usize = @intCast(arr.count);
        const strings = try allocator.alloc([]const u8, count);

        var pos: usize = 0;
        for (0..count) |i| {
            if (pos + 8 > arr.raw_data.len) return error.UnexpectedEof;
            const str_len: usize = @intCast(readU64(arr.raw_data[pos..][0..8]));
            pos += 8;
            if (pos + str_len > arr.raw_data.len) return error.UnexpectedEof;
            strings[i] = arr.raw_data[pos..][0..str_len];
            pos += str_len;
        }
        return strings;
    }

    /// モデル情報をデバッグ表示
    pub fn printInfo(self: *const GGUFFile) void {
        std.debug.print("GGUF v{d}: {d} tensors, {d} metadata KV\n", .{
            self.version, self.tensors.len, self.metadata.len,
        });

        if (self.getMetadataString("general.architecture")) |arch| {
            std.debug.print("  Architecture: {s}\n", .{arch});
        }
        if (self.getMetadataString("general.name")) |name| {
            std.debug.print("  Name: {s}\n", .{name});
        }

        std.debug.print("  Tensors:\n", .{});
        for (self.tensors) |t| {
            std.debug.print("    {s}: ", .{t.name});
            for (0..t.n_dims) |d| {
                if (d > 0) std.debug.print("x", .{});
                std.debug.print("{d}", .{t.dimensions[d]});
            }
            std.debug.print(" (type={d})\n", .{@intFromEnum(t.type_)});
        }
    }
};

// ============================================================
// バイナリ読み込みユーティリティ
// ============================================================

fn readU32(buf: *const [4]u8) u32 {
    return std.mem.readInt(u32, buf, .little);
}

fn readU64(buf: *const [8]u8) u64 {
    return std.mem.readInt(u64, buf, .little);
}

fn readI32(buf: *const [4]u8) i32 {
    return std.mem.readInt(i32, buf, .little);
}

// ============================================================
// GGUF パーサー
// ============================================================

pub fn parse(allocator: Allocator, io: std.Io, file_path: []const u8) !GGUFFile {
    // ファイル全体を読み込み
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, file_path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const file_size = stat.size;
    const file_buf = try allocator.alloc(u8, file_size);
    errdefer allocator.free(file_buf);

    const bytes_read = try file.readPositionalAll(io, file_buf, 0);
    if (bytes_read != file_size) return error.IncompleteRead;

    return parseFromBuffer(allocator, file_buf, file_path);
}

pub fn parseFromBuffer(allocator: Allocator, file_buf: []const u8, file_path: []const u8) !GGUFFile {
    var pos: usize = 0;

    // Header
    if (file_buf.len < 20) return error.FileTooSmall;

    const magic = readU32(file_buf[pos..][0..4]);
    pos += 4;
    if (magic != GGUF_MAGIC) return error.InvalidMagic;

    const version = readU32(file_buf[pos..][0..4]);
    pos += 4;
    if (version < 2 or version > 3) return error.UnsupportedVersion;

    const tensor_count: usize = @intCast(readU64(file_buf[pos..][0..8]));
    pos += 8;
    const metadata_kv_count: usize = @intCast(readU64(file_buf[pos..][0..8]));
    pos += 8;

    // Metadata
    const metadata = try allocator.alloc(MetadataKV, metadata_kv_count);
    errdefer allocator.free(metadata);

    for (0..metadata_kv_count) |i| {
        const kv = try readMetadataKV(file_buf, &pos);
        metadata[i] = kv;
    }

    // アライメント値を取得
    var alignment: u64 = 32;
    for (metadata) |kv| {
        if (std.mem.eql(u8, kv.key, "general.alignment")) {
            alignment = switch (kv.value) {
                .uint32 => |v| v,
                .uint64 => |v| v,
                else => 32,
            };
            break;
        }
    }

    // Tensor info
    const tensors = try allocator.alloc(TensorInfo, tensor_count);
    errdefer allocator.free(tensors);

    for (0..tensor_count) |i| {
        tensors[i] = try readTensorInfo(file_buf, &pos);
    }

    // データセクション開始 = 現在位置をアラインメント境界に切り上げ
    const align_u: usize = @intCast(alignment);
    const data_offset = (pos + align_u - 1) / align_u * align_u;

    // file_path をコピー
    const path_copy = try allocator.alloc(u8, file_path.len);
    @memcpy(path_copy, file_path);

    return .{
        .version = version,
        .metadata = metadata,
        .tensors = tensors,
        .data_offset = @intCast(data_offset),
        .alignment = alignment,
        .file_path = path_copy,
        .file_buf = file_buf,
        .allocator = allocator,
    };
}

fn readGGUFString(buf: []const u8, pos: *usize) ![]const u8 {
    if (pos.* + 8 > buf.len) return error.UnexpectedEof;
    const str_len: usize = @intCast(readU64(buf[pos.*..][0..8]));
    pos.* += 8;
    if (pos.* + str_len > buf.len) return error.UnexpectedEof;
    const str = buf[pos.*..][0..str_len];
    pos.* += str_len;
    return str;
}

fn readMetadataKV(buf: []const u8, pos: *usize) !MetadataKV {
    const key = try readGGUFString(buf, pos);

    if (pos.* + 4 > buf.len) return error.UnexpectedEof;
    const value_type_raw = readU32(buf[pos.*..][0..4]);
    pos.* += 4;

    const value = try readMetadataValue(buf, pos, value_type_raw);

    return .{ .key = key, .value = value };
}

fn readMetadataValue(buf: []const u8, pos: *usize, value_type_raw: u32) !MetadataValue {
    return switch (@as(ValueType, @enumFromInt(value_type_raw))) {
        .uint8 => blk: {
            if (pos.* + 1 > buf.len) return error.UnexpectedEof;
            const v = buf[pos.*];
            pos.* += 1;
            break :blk .{ .uint8 = v };
        },
        .int8 => blk: {
            if (pos.* + 1 > buf.len) return error.UnexpectedEof;
            const v: i8 = @bitCast(buf[pos.*]);
            pos.* += 1;
            break :blk .{ .int8 = v };
        },
        .uint16 => blk: {
            if (pos.* + 2 > buf.len) return error.UnexpectedEof;
            const v = std.mem.readInt(u16, buf[pos.*..][0..2], .little);
            pos.* += 2;
            break :blk .{ .uint16 = v };
        },
        .int16 => blk: {
            if (pos.* + 2 > buf.len) return error.UnexpectedEof;
            const v = std.mem.readInt(i16, buf[pos.*..][0..2], .little);
            pos.* += 2;
            break :blk .{ .int16 = v };
        },
        .uint32 => blk: {
            if (pos.* + 4 > buf.len) return error.UnexpectedEof;
            const v = readU32(buf[pos.*..][0..4]);
            pos.* += 4;
            break :blk .{ .uint32 = v };
        },
        .int32 => blk: {
            if (pos.* + 4 > buf.len) return error.UnexpectedEof;
            const v = readI32(buf[pos.*..][0..4]);
            pos.* += 4;
            break :blk .{ .int32 = v };
        },
        .float32 => blk: {
            if (pos.* + 4 > buf.len) return error.UnexpectedEof;
            const v: f32 = @bitCast(buf[pos.*..][0..4].*);
            pos.* += 4;
            break :blk .{ .float32 = v };
        },
        .bool_ => blk: {
            if (pos.* + 1 > buf.len) return error.UnexpectedEof;
            const v = buf[pos.*] != 0;
            pos.* += 1;
            break :blk .{ .bool_ = v };
        },
        .string => blk: {
            const s = try readGGUFString(buf, pos);
            break :blk .{ .string = s };
        },
        .array => blk: {
            if (pos.* + 12 > buf.len) return error.UnexpectedEof;
            const elem_type: ValueType = @enumFromInt(readU32(buf[pos.*..][0..4]));
            pos.* += 4;
            const count = readU64(buf[pos.*..][0..8]);
            pos.* += 8;

            // 配列の生バイトを記録（要素を実際にパースしてスキップする）
            const raw_start = pos.*;
            for (0..count) |_| {
                _ = try readMetadataValue(buf, pos, @intFromEnum(elem_type));
            }
            const raw_end = pos.*;

            break :blk .{ .array = .{
                .elem_type = elem_type,
                .count = count,
                .raw_data = buf[raw_start..raw_end],
            } };
        },
        .uint64 => blk: {
            if (pos.* + 8 > buf.len) return error.UnexpectedEof;
            const v = readU64(buf[pos.*..][0..8]);
            pos.* += 8;
            break :blk .{ .uint64 = v };
        },
        .int64 => blk: {
            if (pos.* + 8 > buf.len) return error.UnexpectedEof;
            const v = std.mem.readInt(i64, buf[pos.*..][0..8], .little);
            pos.* += 8;
            break :blk .{ .int64 = v };
        },
        .float64 => blk: {
            if (pos.* + 8 > buf.len) return error.UnexpectedEof;
            const v: f64 = @bitCast(buf[pos.*..][0..8].*);
            pos.* += 8;
            break :blk .{ .float64 = v };
        },
        _ => return error.UnsupportedValueType,
    };
}

fn readTensorInfo(buf: []const u8, pos: *usize) !TensorInfo {
    const name = try readGGUFString(buf, pos);

    if (pos.* + 4 > buf.len) return error.UnexpectedEof;
    const n_dims = readU32(buf[pos.*..][0..4]);
    pos.* += 4;

    if (n_dims > 4) return error.TooManyDimensions;

    var dims: [4]u64 = .{ 0, 0, 0, 0 };
    for (0..n_dims) |i| {
        if (pos.* + 8 > buf.len) return error.UnexpectedEof;
        dims[i] = readU64(buf[pos.*..][0..8]);
        pos.* += 8;
    }

    if (pos.* + 4 > buf.len) return error.UnexpectedEof;
    const type_raw = readU32(buf[pos.*..][0..4]);
    pos.* += 4;

    if (pos.* + 8 > buf.len) return error.UnexpectedEof;
    const offset = readU64(buf[pos.*..][0..8]);
    pos.* += 8;

    return .{
        .name = name,
        .n_dims = n_dims,
        .dimensions = dims,
        .type_ = @enumFromInt(type_raw),
        .offset = offset,
    };
}

// ============================================================
// テスト
// ============================================================

test "parse minimal GGUF" {
    // 手作りの最小 GGUF バイナリ:
    // magic(4) + version(4) + tensor_count(8) + kv_count(8)
    // + 1 metadata KV (key="test.key", type=uint32, value=42)
    // + 1 tensor info (name="w", n_dims=1, dims=[4], type=f32, offset=0)
    // + padding + tensor data (4 x f32)
    const alloc = std.testing.allocator;

    var buf: [256]u8 = undefined;
    var pos: usize = 0;

    // Header
    writeU32(&buf, &pos, GGUF_MAGIC);
    writeU32(&buf, &pos, 3); // version
    writeU64(&buf, &pos, 1); // tensor_count
    writeU64(&buf, &pos, 1); // kv_count

    // Metadata KV: key="test.key", uint32, 42
    writeString(&buf, &pos, "test.key");
    writeU32(&buf, &pos, 4); // uint32 type
    writeU32(&buf, &pos, 42);

    // Tensor info: name="w", 1 dim, [4], f32, offset=0
    writeString(&buf, &pos, "w");
    writeU32(&buf, &pos, 1); // n_dims
    writeU64(&buf, &pos, 4); // dim[0]
    writeU32(&buf, &pos, 0); // f32
    writeU64(&buf, &pos, 0); // offset

    // Pad to 32 byte alignment
    const data_start = (pos + 31) / 32 * 32;
    @memset(buf[pos..data_start], 0);

    // Tensor data: 4 f32s
    const vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const data_bytes = std.mem.sliceAsBytes(&vals);
    @memcpy(buf[data_start..][0..data_bytes.len], data_bytes);

    const total_len = data_start + data_bytes.len;

    // パース (バッファのコピーを渡す)
    const buf_copy = try alloc.alloc(u8, total_len);
    @memcpy(buf_copy, buf[0..total_len]);

    var gguf_file = try parseFromBuffer(alloc, buf_copy, "test.gguf");
    defer gguf_file.deinit();

    // 検証
    try std.testing.expectEqual(@as(u32, 3), gguf_file.version);
    try std.testing.expectEqual(@as(usize, 1), gguf_file.metadata.len);
    try std.testing.expectEqual(@as(usize, 1), gguf_file.tensors.len);

    // Metadata
    const val = gguf_file.getMetadataU32("test.key");
    try std.testing.expectEqual(@as(?u32, 42), val);

    // Tensor info
    const t = gguf_file.getTensorInfo("w").?;
    try std.testing.expectEqual(@as(u32, 1), t.n_dims);
    try std.testing.expectEqual(@as(u64, 4), t.dimensions[0]);
    try std.testing.expectEqual(GGMLType.f32, t.type_);

    // Load tensor
    const data = try gguf_file.loadTensorF32("w", alloc);
    defer alloc.free(data);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), data[3], 1e-6);
}

// テスト用のバイナリ書き込みヘルパー
fn writeU32(buf: []u8, pos: *usize, val: u32) void {
    std.mem.writeInt(u32, buf[pos.*..][0..4], val, .little);
    pos.* += 4;
}

fn writeU64(buf: []u8, pos: *usize, val: u64) void {
    std.mem.writeInt(u64, buf[pos.*..][0..8], val, .little);
    pos.* += 8;
}

fn writeString(buf: []u8, pos: *usize, str: []const u8) void {
    writeU64(buf, pos, str.len);
    @memcpy(buf[pos.*..][0..str.len], str);
    pos.* += str.len;
}
