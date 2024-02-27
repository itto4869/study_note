reset
set terminal tikz
set output "prml/ch1/fig/rms_error.tex"
set xlabel "$M$"
set ylabel "$E_{RMS}$"
set xrange [-0.2:9.2]
set yrange [0:1]
plot "prml/ch1/fig/data/rms_error.txt" using 1:2 with lines title "Training", \
     "prml/ch1/fig/data/rms_error.txt" using 1:3 with lines title "Test"