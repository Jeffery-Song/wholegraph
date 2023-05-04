#!/usr/bin/env gnuplot

dat_file="data.dat"
set datafile sep '\t'
set output "figure.svg"
# set terminal postscript "Helvetica,16" eps enhance color dl 2
set terminal svg

set pointsize 1
# set size 0.4,0.35
set size 1.2,1.1
# set zeroaxis

set multiplot layout 3,3

set tics font ",14" scale 0.5

set rmargin 0 #2
set lmargin 5 #5.5
set tmargin 0.5 #1.5
set bmargin 1 #2.5

### Key
set key inside left Left top enhanced nobox  autotitles columnhead reverse
set key samplen 1.5 spacing 1.5 height 0.2 width 0 font ',13' center at graph 0.68, graph 0.85 #maxrows 1 at graph 0.02, graph 0.975  noopaque
# unset key

## Y-axis
set ylabel "Hit Rate(%)" offset 2.,0
# set logscale y
set yrange [0:]
# set ytics (0.5,0.6,0.8,1,2,4)
set ytics offset 0.5,0 #format "%.1f" #nomirror


### X-axis
set xlabel "Cache Rate" offset 0,0.7
# set xrange [0.9:2.1]
#set xtics 1,1,8 
set xtics nomirror offset 0,0.3

set grid ytics dashtype(10,10) lw 1 lc "#222222"

# set label 3 "Theory" center at graph 0.5, first 144 font ",18" tc rgb "#000000" front
# set arrow 1 from graph 0, first 144 to  graph 1, first 144 nohead lt 1 lw 3 dashtype(3,2) lc "#000000"
# set arrow 2 from graph 0.38, first 144 to  graph 0.63, first 144 nohead lt 1 lw 5 lc "#ffffff"
# set arrow from   8, graph -0.3 to  8, graph 1.0 nohead lt 1 lw 2 lc "#000000" front
# set arrow from  15, graph -0.3 to 15, graph 1.0 nohead lt 1 lw 2 lc "#000000" front
# set arrow from  21, graph -0.3 to 21, graph 0.0 nohead lt 1 lw 2 lc "#000000" front

# set datafile missing "-"

do for [block_select=0:8]  {


plot dat_file every 3::1:block_select::block_select using 3:($6)      w lp lw 3 lc rgb "#0000ee" t 'Rep', \
     dat_file every 3::0:block_select::block_select using 3:($6)      w lp lw 3 lc rgb "#c00000" t 'Part.Local', \
     dat_file every 3::0:block_select::block_select using 3:($6 + $7) w lp lw 3 lc rgb "#c00000" t 'Part.Global', \
    #  dat_file every :::0::0 using 3:($6*100) w lp lw 3 lc rgb "#c00000"
}
# plot "scale-break.dat" using ($5):xticlabels(1) t "GNNLab"  w lp lt 1 lw 3 pt 6 ps 1.5 lc rgb '#000000', \
#      "scale-break.dat" using ($2):xticlabels(1) t "Sample"  w histogram lc rgb "#ff9900",\
#      "scale-break.dat" using ($3):xticlabels(1) t "Extract" w histogram lc rgb "#c00000", \
#      "scale-break.dat" using ($4):xticlabels(1) t "Train"   w histogram lc rgb "#0000ee", \

##008800
