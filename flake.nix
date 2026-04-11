{
  description = "Zig NN library";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forAllSystems ({ pkgs }: {
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
            pkgs.openblas
          ];
          shellHook = ''
            export NIX_CFLAGS_COMPILE=""
            export ZIG_GLOBAL_CACHE_DIR=/tmp/zig-cache
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export OPENBLAS_PATH="${pkgs.openblas}"
          '';
        };
      });
    };
}
