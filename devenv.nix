{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = [ pkgs.git 
    pkgs.go
    pkgs.gcc11
    pkgs.gomp
    pkgs.templ
    pkgs.uv
    
    pkgs.ruff
    pkgs.python311Full
    pkgs.python311Packages.pip
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.scipy
    pkgs.python311Packages.numpy
    pkgs.python311Packages.tensorflow
    pkgs.python311Packages.trimesh
    pkgs.python311Packages.numpy-stl

     pkgs.primesieve         # primesieve library
      pkgs.pkg-config         # pkg-config to help find primesieve
      pkgs.boost
    
      pkgs.cudaPackages_12.cudatoolkit 
  ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
 languages.javascript = {
    enable = true; # adds node LTS & npm
    package = pkgs.nodejs_18;
  };

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  processes = {
    tailwind.exec = "cd web/tailwind && npm run watch-css";
    air.exec = "air";
    temple.exec = "cd server/views && ls -l && templ generate --watch -v";
  };

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts={
    gitst.exec = ''
      git status
    '';
    gitco.exec = ''
      git commit -m "$@"
    '';
    gitadd.exec = ''
      git add . 
    '';
    gitad.exec = ''
      git add . 
    '';
    dev.exec = ''
      devenv "$@"
    '';
    gitlog.exec = ''
      git log --oneline
    '';
    lg1.exec = ''
      git lg1
    '';
    lg2.exec = ''
      git lg2
    '';
    buildw.exec = ''
      GOOS=windows GOARCH=amd64 go build
    '';
    build.exec = ''
      go build
    '';
    compileS4.exec = ''
      nvcc -arch=sm_60 -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_86,code=sm_86 \
 -o s4 s4.cu  /home/darrell/work/git/primesieve/libprimesieve.a -std=c++14 --expt-relaxed-constexpr -Xcompiler="-fpermissive" -lnvidia-ml
      patchelf --set-rpath '/run/opengl-driver/lib:'$(patchelf --print-rpath s4) s4
      patchelf --add-rpath /usr/lib/x86_64-linux-gnu s4
     '';
    compileS3.exec = ''
      nvcc -arch=sm_86 -o s3 s3.cu /home/darrell/work/git/primesieve/libprimesieve.a -std=c++14 --expt-relaxed-constexpr -Xcompiler="-fpermissive" -lnvidia-ml
      patchelf --set-rpath '/run/opengl-driver/lib:'$(patchelf --print-rpath s3) s3
      '';
  };

  #process-managers.overmind.enable= lib.mkOptionDefault true;

  enterShell = ''
    hello
    unset __HM_SESS_VARS_SOURCED ; . ~/.profile;
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
