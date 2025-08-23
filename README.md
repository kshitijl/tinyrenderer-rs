## what is it

Pure CPU software renderer written in Rust, following [Dmitry V. Sokolov's tinyrenderer course](https://github.com/ssloy/tinyrenderer/wiki).

## show me the good stuff! SHOW ME the GOOOOOD stuff it makes

Here is the journey of this renderer so far:

### Test checkerboard patterns to see that TGA output works
![](./renders/01.png)
![](./renders/02.png)

### Drawing points and lines, working up to Bresenham's all-integer line-drawing algorithm
![](./renders/03.png)
![](./renders/04.png)
![](./renders/05.png)
![](./renders/06.png)
![](./renders/07-bresenham.png)

### Loading and drawing a model by simply dropping the Z coordinates and drawing triangles

#### These are buggy! I had bugs in my Bresenham code
![](./renders/08.png)
![](./renders/09.png)
![](./renders/10.png)

#### This is correct
![](./renders/11-diablo.png)
