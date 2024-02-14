reset
set terminal tikz
set output "prml/ch1/fig/toy_data.tex"
set title "Toy Dataset"
set xlabel "$x$"
set ylabel "$t$"
set xrange [-0.1:1.1]
set yrange [-1.2:1.2]
f(x) = sin(2*pi*x)
plot f(x) title "$\\sin(2\\pi x)$" linecolor "green", \
        "prml/ch1/fig/data/toy_data.txt" using 1:2 with points pointtype 7 linecolor "blue" title "$\\sin(2\\pi x) + noise$"