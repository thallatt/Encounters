#! /usr/bin/env gnuplot 

set fontpath "/Users/Tim/Library/texmf/fonts/type1/public/cm-super/"
set datafile separator ","
set terminal postscript eps 14 font "SFRM1000" fontfile "sfrm1000.pfb" enhanced color
set output "clones.eps"

set lmargin at screen 0.17
set rmargin at screen 0.9
set tmargin at screen 0.9
set bmargin at screen 0.17

xmin = 5
xmax = 26
set xrange [xmin:xmax]
ymin = 0
ymax = 3.5
set yrange [ymin:ymax]

set mytics 5
set mxtics 5

set xtics border scale "2.5" font ",19"
set ytics border scale "2.5" font ",19"

set xlabel "Relative Speed [km/s]" offset 0,-1.5 font ",19"
set ylabel "Distance [pc]" offset -1.5,0 font ",19"

set rmargin 8
set border 15 lc rgb "black"

set key off
set bars 0.7

set pm3d map
clr(time) = (65 + -time/10.e6 * 190)*65536 + (65 - -time/10.e6 * 5)*256 + (65 - -time/10.e6 * 65)
set palette defined (0 "black", 1 "light-salmon")
set cblabel "Encounter Time [/t_{end}]" font ",19" offset 1, 0

plot 'candidate_data.csv' using 13:7:9:10:3:4:(-$2 / 10.e6) with xyerrorbars pointtype 7 pointsize 0.5 lw 2 lc palette z notitle
