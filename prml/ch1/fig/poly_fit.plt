reset
set terminal tikz
set output "prml/ch1/fig/poly_fit.tex"
set multiplot layout 2,2
set xlabel "$x$"
set ylabel "$t$"
set xrange [-0.1:1.1]
set yrange [-1.5:1.5]
f(x) = sin(2*pi*x)
plot "prml/ch1/fig/data/plot_data_poly_fit.txt" using 5:1 with line linecolor "red" title "$M=0$", \
        f(x) with line lt 3 lc "green" notitle, \
        "prml/ch1/fig/data/toy_data.txt" using 1:2 with points pt 7 lc "blue" notitle
plot "prml/ch1/fig/data/plot_data_poly_fit.txt" using 5:2 with line lc "red" title "$M=1$", \
        f(x) with line lt 3 lc "green" notitle, \
        "prml/ch1/fig/data/toy_data.txt" using 1:2 with points pt 7 lc "blue" notitle
plot "prml/ch1/fig/data/plot_data_poly_fit.txt" using 5:3 with line lc "red" title "$M=3$", \
        f(x) with line lt 3 lc "green" notitle, \
        "prml/ch1/fig/data/toy_data.txt" using 1:2 with points pt 7 lc "blue" notitle
plot "prml/ch1/fig/data/plot_data_poly_fit.txt" using 5:4 with line lc "red" title "$M=9$", \
        f(x) with line lt 3 lc "green" notitle, \
        "prml/ch1/fig/data/toy_data.txt" using 1:2 with points pt 7 lc "blue" notitle

unset multiplot