{
  description = "Zig NN library";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        zig_0_16 = let
          zigBin = {
            "aarch64-darwin" = {
              url = "https://ziglang.org/download/0.16.0/zig-aarch64-macos-0.16.0.tar.xz";
              hash = "0yqiq1nrjfawh1k24mf969q1w9bhwfbwqi2x8f9zklca7bsyza26";
            };
            "x86_64-darwin" = {
              url = "https://ziglang.org/download/0.16.0/zig-x86_64-macos-0.16.0.tar.xz";
              hash = "0dibmghlqrr8qi5cqs9n0nl25qdnb5jvr542dyljfqdyy2bzzh2x";
            };
            "x86_64-linux" = {
              url = "https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz";
              hash = "1kgamnyy7vsw5alb5r4xk8nmgvmgbmxkza5hs7b51x6dbgags1h6";
            };
            "aarch64-linux" = {
              url = "https://ziglang.org/download/0.16.0/zig-aarch64-linux-0.16.0.tar.xz";
              hash = "12gf4d1rjncc8r4i32sfdmnwdl0d6hg717hb3801zxjlmzmpsns0";
            };
          }.${system};
          src = builtins.fetchTarball {
            url = zigBin.url;
            sha256 = zigBin.hash;
          };
        in pkgs.stdenv.mkDerivation {
          pname = "zig";
          version = "0.16.0";
          inherit src;
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            cp -r ./* $out/
            mkdir -p $out/bin
            ln -sf $out/zig $out/bin/zig
          '';
        };

        openblas32 = pkgs.openblas.override { blas64 = false; };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          matplotlib
          numpy
          torch
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            zig_0_16
            pythonEnv
            pkgs.git
            pkgs.just
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.apple-sdk_15
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
}
