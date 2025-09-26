{
  description = "Development environment for GloVe.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShells.default = with pkgs;
          mkShell {
            buildInputs = [
              clang-tools
              gnumake
              python312
              python312Packages.numpy
              python312Packages.matplotlib
              python312Packages.scikit-learn
            ];
          };
      }
    );
}
