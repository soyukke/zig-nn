/// オプティマイザ共通型・ユーティリティ。
/// Adam, SGD, RMSProp で共有される Param 型と zeroGrad 関数。

/// パラメータ参照: data と grad のペア
pub fn Param(comptime T: type) type {
    return struct {
        data: []T,
        grad: *?[]T,
    };
}

/// 全パラメータの勾配をゼロにリセット
pub fn zeroGrad(comptime T: type, params: []const Param(T)) void {
    for (params) |p| {
        if (p.grad.*) |g| {
            @memset(g, 0);
        }
    }
}
