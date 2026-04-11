/// Deprecated: use Trainer(ModelType, .cuda) instead
const trainer_mod = @import("trainer.zig");

pub fn CudaTrainer(comptime ModelType: type) type {
    return trainer_mod.Trainer(ModelType, .cuda);
}
