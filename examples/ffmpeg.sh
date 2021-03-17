 ffmpeg -i 0Q914by5A98\#010440\#010764.mp4-DMEaUoA8EPE\#000028\#000354.mp4.mp4 -vf "fps=30,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 first.gif

