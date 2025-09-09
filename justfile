launch:
    RUST_BACKTRACE=full RUST_LOG=info cargo run --release -- -c 320 -e assets/cannon.obj -e assets/rhino.obj -e assets/skull.obj -e assets/toilet.obj -e assets/diablo.obj -e assets/giraffe.obj -e assets/greek-vase.obj -e assets/guitar.obj -e assets/monstera_plant_medium_potted.obj -e assets/bat.obj -e assets/celestial-globe.obj -e assets/elephant.obj -e assets/anchor.obj -e assets/walrus.obj --num-guards 20

profile:
    samply record -- RUST_BACKTRACE=full RUST_LOG=info cargo run --release -- -c 320
