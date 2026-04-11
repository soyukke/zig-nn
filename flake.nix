{
  description = "Zig NN library";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f {
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      });
    in
    {
      devShells = forAllSystems ({ pkgs }:
        let
          openblas32 = pkgs.openblas.override { blas64 = false; };
        in {
        default = pkgs.mkShell {
          packages = with pkgs; [
            git
            zig
            (python3.withPackages (ps: with ps; [
              matplotlib
              numpy
              torch
            ]))
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            openblas32.dev
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.cuda_nvcc
            pkgs.cudaPackages.libcublas.lib
          ];
          shellHook = ''
            export NIX_CFLAGS_COMPILE=""
            export ZIG_GLOBAL_CACHE_DIR=/tmp/zig-cache
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export OPENBLAS_PATH="${openblas32}"
            export CUDA_PATH="${pkgs.cudaPackages.cuda_cudart}"
            export CUBLAS_PATH="${pkgs.cudaPackages.libcublas.lib}"
          '';
        };
      });
    };
}
