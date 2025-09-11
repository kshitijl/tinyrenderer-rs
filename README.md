## what is it

Pure CPU software renderer written in Rust, following [Dmitry V. Sokolov's tinyrenderer course](https://github.com/ssloy/tinyrenderer/wiki).

## can I run it?

Yes! I mean probably. It works on my 2021 M1 MacBook Pro. I might have hardcoded things that won't work on other systems.

You'll need the [rust toolchain](https://rustup.rs/) and [just](https://github.com/casey/just) installed. Then:

```
git clone https://github.com/kshitijl/tinyrenderer-rs.git
cd tinyrenderer-rs
just launch
```

### How do I install just?

After installing rust, do `cargo install just`.

## show me the good stuff! SHOW ME the GOOOOOD stuff it makes

Here's a teaser of the end state:

![](./renders/59-gameplay.webp)

Here is the journey of this renderer so far:

### Test checkerboard patterns to see that TGA output works
![](./renders/01.png)
![](./renders/02.png)

### Drawing points and lines, working up to Bresenham's all-integer line-drawing algorithm

<img src="./renders/03.png" width=500></img>

<img src="./renders/04.png" width=500></img>

<img src="./renders/05.png" width=500></img>

<img src="./renders/06.png" width=500></img>

<img src="./renders/07-bresenham.png" width=500></img>

### Loading and drawing a model by simply dropping the Z coordinates and drawing triangles

#### These are buggy! I had bugs in my Bresenham code
![](./renders/08.png)
![](./renders/09.png)
![](./renders/10.png)

#### This is correct
![](./renders/11-diablo.png)

### You can do rotations in 3D with very little code
![](./renders/12-diablo-rotating.webp)

### Filling in triangles
![](./renders/14.png)
![](./renders/15.png)
![](./renders/17-fill-triangles-one-loop.png)
![](./renders/18-diablo-with-triangles.png)

### Lighting
![](./renders/20-head-lit.png)
![](./renders/21-head-back-faces-lit.png)
![](./renders/22-head-lit.webp)
![](./renders/23-diablo-lit.webp)

![](./renders/24-debug-bounding-boxes.webp)

### You can do shader effects
![](./renders/25-spooky.png)
![](./renders/26-ellipses.png)
![](./renders/27-truss.png)

### Depth
![](./renders/28-sort-by-z.webp)
![](./renders/29-depth-tested-diablo.png)
![](./renders/30-depth-buffer-diablo.png)
![](./renders/31-depth-tested-head.png)
![](./renders/32-depth-buffer-head.png)
![](./renders/33-head-lit.webp)

### Perspective
![](./renders/34-megahead.png)
![](./renders/35-optical-illusion-bad-perspective.webp)
![](./renders/36-perspective-diablo-fixed.webp)
![](./renders/37-perspective-head-fixed.webp)
![](./renders/38-persective-float-depth-buffer-head.webp)
![](./renders/39-perspective-correct-z-interpolation.webp)

### Normals for lighting
![](./renders/40-normal-mapped-head.png)
![](./renders/41-phong-shading-with-quantization.png)
![](./renders/42-transform-normals-correctly.webp)
![](./renders/43-buggy-normal-transform.webp)

### Mouselook
![](./renders/44-mouselook.webp)

### Visualizing the depth buffer for shadow mapping
![](./renders/45-depth-buffer-side-by-side.png)
![](./renders/46-moving-light.webp)
![](./renders/47-lights-pov.png)
![](./renders/48-shadows.webp)
![](./renders/49-shadows.png)


### More bugs
![](./renders/50-clearing-bug.png)
![](./renders/50-wireframe-debug.png)
![](./renders/51-lots-of-objects.png)

### Trying to get a nice looking flashlight
![](./renders/52-harsh-flashlight.png)
![](./renders/53-soft-flashlight.png)
![](./renders/54-chromatic-aberration.png)

### What's the point of a renderer, show me a game already
![](./renders/55-topdown-view.png)
![](./renders/56-debug-grid.png)
![](./renders/57-pixelation-effect.png)
![](./renders/58-pixelation-effect-topdown.png)
